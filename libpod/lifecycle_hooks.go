//go:build !remote

package libpod

import (
	"context"
	"fmt"
	"sync"
)

// LifecycleStage identifies a point in a container's start/stop lifecycle
// where an in-process hook can run.
type LifecycleStage int

const (
	// BeforeStart runs just before the OCI runtime starts the container
	// process. Returning an error aborts the start; the container stays
	// in the Created state.
	BeforeStart LifecycleStage = iota
	// AfterStart runs once the container is marked Running and the
	// "start" event has been recorded. Errors are logged, not fatal:
	// the container is already running by this point.
	AfterStart
	// BeforeStop runs just before podman attempts to stop the container.
	// Returning an error aborts the stop.
	BeforeStop
	// AfterStop runs once the container is confirmed stopped and the
	// "stop" event has been recorded. Errors are logged, not fatal.
	//
	// AfterStop is NOT guaranteed to run for every container. conmon's
	// exit-command mechanism (see fullCleanup in container_internal.go)
	// races an explicit "podman stop" to tear the container down after
	// its process dies: if that race resolves in the cleanup process's
	// favor -- likely for --rm containers, since fullCleanup also removes
	// them -- stopInternal's own syncContainer call sees ErrNoSuchCtr/
	// ErrCtrRemoved and returns before ever reaching AfterStop (see the
	// early return in stopInternal). A hook that must not miss a
	// container's teardown should also register on AfterCleanup, which
	// fullCleanup calls directly and unconditionally.
	AfterStop
	// AfterCleanup runs at the end of fullCleanup, which every container
	// goes through exactly once after it stops (via conmon's registered
	// exit-command, regardless of --rm) to unmount storage and tear down
	// its network namespace. Unlike AfterStop, this is not skipped by the
	// stop/cleanup race described above -- it runs from inside the
	// cleanup path itself. Errors are logged, not fatal.
	//
	// This commonly fires for the same container transition AfterStop
	// does (a normal "podman stop" is itself followed by conmon's
	// exit-command running fullCleanup), so a hook registered on both
	// stages should be idempotent -- expect to be called twice for most
	// containers and exactly once for containers where AfterStop lost
	// the race.
	AfterCleanup
)

// LifecycleHookInfo is a read-only snapshot of container data handed to a
// hook. It is a copy, not a live reference: hooks run with c.lock held, so
// they must never call back into a Container method that itself takes the
// lock (that includes most exported Container methods).
type LifecycleHookInfo struct {
	ID     string
	Name   string
	Image  string
	PodID  string
	Labels map[string]string
	Args   []string

	// Annotations are the container's OCI annotations, e.g.
	// {"SLURM_JOB_ID": "42"} from --annotation SLURM_JOB_ID=42. Unlike
	// reading os.Getenv in a hook body, this reflects what was actually
	// recorded on *this* container at create time, regardless of which
	// process (and which environment) later triggers a hook on it --
	// e.g. AfterStop, which runs during whatever separate "podman stop"
	// invocation happens to stop it, possibly with no SLURM_JOB_ID in
	// its own environment, or a different one entirely.
	Annotations map[string]string

	// CDIDevices lists the CDI-qualified devices requested for this
	// container, e.g. "nvidia.com/gpu=all" from --gpus all or
	// "nvidia.com/gpu=0" from --device=nvidia.com/gpu=0. Empty if none
	// were requested. Populated at every stage (it comes from container
	// config, not runtime state).
	CDIDevices []string

	// DevicePaths lists the raw host device paths requested for this
	// container, e.g. "/dev/nvidia0" from --device=/dev/nvidia0. These
	// are the non-CDI entries: anything qualified enough to be a CDI
	// device (like nvidia.com/gpu=0) ends up in CDIDevices instead.
	// Populated at every stage.
	DevicePaths []string

	// PID and ConmonPID are the last known values for this container.
	// Both are 0 at BeforeStart (nothing has run yet); once the
	// container has started they stay populated through BeforeStop and
	// AfterStop, even after the process has exited.
	PID       int
	ConmonPID int

	// CgroupPath is only set while the container is Running or Paused,
	// which in practice means: empty at BeforeStart, populated at
	// AfterStart and BeforeStop, empty again at AfterStop (the state has
	// already moved to Stopping by then). See Container.CgroupPath for
	// the caveats on what this path means under cgroups v1 vs v2.
	CgroupPath string
}

// LifecycleHook is a function compiled into the podman binary that runs at
// a container lifecycle boundary, in-process -- no exec, no external
// script. It complements the file-based hooks under
// /etc/containers/pre-exec-hooks, which run once per podman invocation,
// before any container-specific data even exists.
type LifecycleHook func(ctx context.Context, info *LifecycleHookInfo) error

var (
	lifecycleHooksMu sync.RWMutex
	lifecycleHooks   = map[LifecycleStage][]LifecycleHook{}
)

// RegisterLifecycleHook registers fn to run at stage, for every container,
// in every podman process built from a tree that links this file. There is
// no dynamic loading involved -- the hook is compiled in, so registering it
// means adding a call to this function (typically from an init() in a
// small file of your own next to this one) and rebuilding podman.
func RegisterLifecycleHook(stage LifecycleStage, fn LifecycleHook) {
	lifecycleHooksMu.Lock()
	defer lifecycleHooksMu.Unlock()
	lifecycleHooks[stage] = append(lifecycleHooks[stage], fn)
}

// runLifecycleHooks invokes every hook registered for stage, in
// registration order, stopping at the first error.
//
// Callers must hold c.lock (start() and stopInternal() already do), so
// hooks must not call back into any Container method that takes the lock.
func (c *Container) runLifecycleHooks(ctx context.Context, stage LifecycleStage) error {
	lifecycleHooksMu.RLock()
	hooks := lifecycleHooks[stage]
	lifecycleHooksMu.RUnlock()
	if len(hooks) == 0 {
		return nil
	}

	info := &LifecycleHookInfo{
		ID:         c.ID(),
		Name:       c.Name(),
		Image:      c.config.RootfsImageName,
		PodID:      c.PodID(),
		Labels:     c.Labels(),
		PID:        c.state.PID,
		ConmonPID:  c.state.ConmonPID,
		CDIDevices: c.config.CDIDevices,
	}
	if c.config.Spec != nil && c.config.Spec.Process != nil {
		info.Args = c.config.Spec.Process.Args
	}
	if c.config.Spec != nil {
		info.Annotations = c.config.Spec.Annotations
	}
	if c.config.Spec != nil && c.config.Spec.Linux != nil {
		for _, d := range c.config.Spec.Linux.Devices {
			info.DevicePaths = append(info.DevicePaths, d.Path)
		}
	}
	// cGroupPath (unexported) is safe to call here: it only reads state
	// and requires the caller to already hold c.lock, which start() and
	// stopInternal() do. It errors when the container isn't Running or
	// Paused, which is expected and fine at BeforeStart/BeforeStop.
	if path, err := c.cGroupPath(); err == nil {
		info.CgroupPath = path
	}

	for _, fn := range hooks {
		if err := fn(ctx, info); err != nil {
			return fmt.Errorf("container %s: lifecycle hook: %w", c.ID(), err)
		}
	}
	return nil
}
