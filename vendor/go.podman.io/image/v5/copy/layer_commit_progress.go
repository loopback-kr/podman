package copy

import (
	"fmt"
	"io"
	"sync"
	"sync/atomic"

	"github.com/vbauerster/mpb/v8"
	"github.com/vbauerster/mpb/v8/decor"
	"go.podman.io/image/v5/internal/private"
	"go.podman.io/image/v5/types"
)

// layerCommitPhase is the phase of a layer being copied to a destination which commits layers
// after receiving their blobs (e.g. c/storage, which extracts them).
type layerCommitPhase int32

const (
	layerPhaseDownloading layerCommitPhase = iota // The blob is being received
	layerPhaseWaiting                             // The blob was received, the layer is waiting to be committed
	layerPhaseCommitting                          // The layer is being committed
)

// trackedLayer is a layer whose commit progress is reported by layerCommitTracker.
type trackedLayer struct {
	info  types.BlobInfo
	bar   *progressBar // nil if progress bars are not rendered
	phase atomic.Int32 // layerCommitPhase

	// The following fields are protected by layerCommitTracker.mu.
	downloaded bool // The caller has finished receiving the blob, and handed the bar over to the tracker
	committed  bool // The destination has reported that the layer has been committed
	printed    bool // An "Extracting blob" line has been printed (if progress bars are not rendered)
	finished   bool // The bar has been completed
}

// layerCommitTracker shows the progress of committing layers into a destination, after their blobs
// have been received, so that the output does not appear to be stuck while a large layer is extracted.
//
// Layers are committed in order, by whichever goroutine happens to be copying the next layer, so a
// layer might be committed before or after its own blob copy returns. The tracker therefore completes
// a layer's progress bar when the destination reports the layer as committed, takes over the bar
// once its blob has been received, and completes any remaining bars in finish().
type layerCommitTracker struct {
	c *copier

	mu     sync.Mutex
	closed bool
	layers map[int]*trackedLayer
}

func newLayerCommitTracker(c *copier) *layerCommitTracker {
	return &layerCommitTracker{
		c:      c,
		layers: map[int]*trackedLayer{},
	}
}

// createProgressBar creates a progress bar for a layer which will be received and then committed.
// Unlike copier.createProgressBar, the bar does not complete when the whole blob has been received;
// the caller must call blobReceived on success, and is responsible for aborting the bar otherwise.
// info.Size must be > 0.
func (t *layerCommitTracker) createProgressBar(pool *mpb.Progress, info types.BlobInfo, layerIndex int) (*progressBar, error) {
	if err := info.Digest.Validate(); err != nil { // digest.Digest.Encoded() panics on failure, so validate explicitly.
		return nil, err
	}
	tl := &trackedLayer{info: info}

	prefix := progressBarPrefix("blob", info)
	size := info.Size
	// The total is one more than the blob size, so that receiving the whole blob does not complete the bar;
	// finishLocked() completes it.
	bar := pool.AddBar(size+1,
		mpb.BarFillerClearOnComplete(),
		mpb.PrependDecorators(
			decor.OnComplete(decor.Name(prefix), prefix+" done"),
		),
		mpb.AppendDecorators(
			decor.OnComplete(decor.Any(func(s decor.Statistics) string {
				current := min(s.Current, size)
				switch layerCommitPhase(tl.phase.Load()) {
				case layerPhaseWaiting:
					return "waiting to extract"
				case layerPhaseCommitting:
					return fmt.Sprintf("extracting %.1f / %.1f", decor.SizeB1024(current), decor.SizeB1024(size))
				default:
					return fmt.Sprintf("%.1f / %.1f | ", decor.SizeB1024(current), decor.SizeB1024(size))
				}
			}), ""),
			decor.OnComplete(phaseDecorator{
				Decorator: decor.EwmaSpeed(decor.SizeB1024(0), "% .1f", 30),
				tl:        tl,
				phase:     layerPhaseDownloading,
			}, ""),
		),
	)
	tl.bar = &progressBar{Bar: bar, originalSize: size}

	t.mu.Lock()
	t.layers[layerIndex] = tl
	t.mu.Unlock()
	return tl.bar, nil
}

// registerWithoutBar registers a layer which will be received and then committed, when progress bars are not rendered.
func (t *layerCommitTracker) registerWithoutBar(info types.BlobInfo, layerIndex int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.layers[layerIndex] = &trackedLayer{info: info}
}

// blobReceived records that the blob for layerIndex has been fully received.
// If it returns true, the tracker has taken over the layer's progress bar, and the caller must not abort it.
// It always returns true for bars created by t.createProgressBar, which the caller must not complete on its own.
func (t *layerCommitTracker) blobReceived(layerIndex int) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	tl, ok := t.layers[layerIndex]
	if !ok || tl.bar == nil {
		return false
	}
	tl.downloaded = true
	if tl.committed || t.closed {
		t.finishLocked(tl)
		return true
	}
	if layerCommitPhase(tl.phase.Load()) == layerPhaseDownloading {
		tl.phase.Store(int32(layerPhaseWaiting))
	}
	return true
}

// report is a callback for private.LayerCommitProgressReporter.
func (t *layerCommitTracker) report(p private.LayerCommitProgress) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return
	}
	tl, ok := t.layers[p.LayerIndex]
	if !ok {
		return
	}
	if p.Done {
		tl.committed = true
		// A layer can only be committed after its whole blob has been received, so the bar can be completed
		// even if the blob copy has not returned yet (it may be committing later layers in the meantime).
		// blobReceived will still hand the bar over to the tracker, so that the caller does not abort it.
		t.finishLocked(tl)
		return
	}

	if tl.bar == nil {
		if !tl.printed {
			tl.printed = true
			t.c.Printf("Extracting blob %s\n", tl.info.Digest)
		}
		return
	}
	if tl.finished {
		return
	}
	tl.phase.Store(int32(layerPhaseCommitting))
	// Scale the commit progress to the bar, which tracks the blob size.
	var current int64
	if p.Size > 0 {
		current = min(p.Offset, p.Size) * tl.info.Size / p.Size
	}
	tl.bar.SetCurrent(current)
}

// finishLocked completes tl's progress bar. t.mu must be held.
func (t *layerCommitTracker) finishLocked(tl *trackedLayer) {
	if tl.finished || tl.bar == nil {
		return
	}
	tl.finished = true
	tl.bar.SetCurrent(tl.info.Size + 1) // This triggers the completion condition.
}

// finish completes all progress bars the tracker has taken over, and stops processing reports.
// It must be called after all layer copies have finished, and before the progress pool is waited on.
// Layers which have not been reported as committed yet will be committed later (e.g. when the image is committed),
// without progress reporting.
func (t *layerCommitTracker) finish() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.closed = true
	for _, tl := range t.layers {
		if tl.downloaded {
			t.finishLocked(tl)
		}
	}
}

// phaseDecorator wraps a decorator, and only shows it while a tracked layer is in a specific phase.
type phaseDecorator struct {
	decor.Decorator
	tl    *trackedLayer
	phase layerCommitPhase
}

func (d phaseDecorator) Decor(s decor.Statistics) (string, int) {
	if layerCommitPhase(d.tl.phase.Load()) != d.phase {
		return d.Format("")
	}
	return d.Decorator.Decor(s)
}

// Unwrap allows mpb to find the wrapped decorator, e.g. to feed EWMA updates to it.
func (d phaseDecorator) Unwrap() decor.Decorator {
	return d.Decorator
}

// layerCommitTrackerFor returns a layerCommitTracker for copying layers to c.dest, and a cleanup function
// to call after all layer copies have finished and before the progress pool is waited on; or (nil, no-op)
// if c.dest does not report layer commit progress.
func (c *copier) layerCommitTrackerFor() (*layerCommitTracker, func()) {
	reporter, ok := c.dest.(private.LayerCommitProgressReporter)
	if !ok || c.reportWriter == io.Discard {
		return nil, func() {}
	}
	t := newLayerCommitTracker(c)
	reporter.SetLayerCommitProgressCallback(t.report)
	return t, func() {
		reporter.SetLayerCommitProgressCallback(nil)
		t.finish()
	}
}
