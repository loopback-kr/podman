//go:build !remote

// Package slurmhooks wires podman's container lifecycle into SLURM.
//
// It is registered as a blank import from cmd/podman/main.go, so it only
// needs to be linked in once; from then on every "podman run/start/stop"
// on this host runs it automatically, in-process, for every container.
package slurmhooks

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/containers/podman/v5/libpod"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
	"go.podman.io/common/pkg/config"
)

const nvidiaCDITimeout = 10 * time.Second

// createSubcommandIndex returns the index of the first bare "run" or
// "create" token in args, or -1 if there isn't one. --gpus, --device and
// --annotation are only ever valid after that point (they're local flags
// registered on run/create, cmd/podman/common/create.go, not persistent
// flags on rootCmd), so both wantsGPU and injectAnnotation need to know
// where podman's own subcommand starts before they can look for them --
// otherwise a token that merely appears earlier or later in argv (e.g.
// as an argument to `podman exec`, or a container/image name) would be
// mistaken for the real thing.
//
// This only recognizes a bare "run" or "create" token -- not aliases
// like "container run".
func createSubcommandIndex(args []string) int {
	for i, a := range args {
		if a == "run" || a == "create" {
			return i
		}
	}
	return -1
}

// wantsGPU is a coarse, argv-only guess at whether this podman invocation
// is requesting a GPU device -- via --gpus or --device after the run/
// create subcommand -- without waiting for full command parsing. It runs
// from init(), before cobra even sees os.Args, at the same point in the
// process lifetime that /etc/containers/pre-exec-hooks scripts would run
// (see pkg/rootless/rootless_linux.c's do_preexec_hooks), just compiled
// into the binary instead of dropped on disk as a script.
//
// This only looks at flag names, not their values, so it doesn't confirm
// the device is actually an NVIDIA one -- rawNvidiaDevicePath and the
// nvidia.com/gpu prefix check elsewhere in this file do that once the
// real container config is available, later in the lifecycle.
func wantsGPU(args []string) bool {
	idx := createSubcommandIndex(args)
	if idx == -1 {
		return false
	}
	for _, a := range args[idx+1:] {
		if a == "--gpus" || strings.HasPrefix(a, "--gpus=") {
			return true
		}
		if a == "--device" || strings.HasPrefix(a, "--device=") {
			return true
		}
	}
	return false
}

// generateNvidiaCDI runs nvidia-ctk before the Go runtime has dispatched
// to any subcommand, so the refreshed CDI spec is on disk well before
// generateSpec() (libpod/container_internal_common.go) reads it during
// container create/init -- the ordering problem a BeforeStart-time call
// could not solve, since generateSpec() already runs before start().
//
// Writes to a per-container path (dir/nvidia.yaml, dir already created by
// the caller) instead of the shared default so concurrent containers --
// even two under the same SLURM job -- don't overwrite each other's spec
// or race over its deletion at stop time.
func generateNvidiaCDI(dir string) {
	ctx, cancel := context.WithTimeout(context.Background(), nvidiaCDITimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-ctk", "cdi", "generate", "--output="+filepath.Join(dir, "nvidia.yaml"))
	if err := cmd.Run(); err != nil {
		logrus.Warnf("CDI generate failed: %v", err)
	}
}

// resolvedCDISpecDirs returns the currently configured CDI spec
// directories (containers.conf's engine.cdi_spec_dirs, or the built-in
// default of /etc/cdi and /var/run/cdi if unset). Falls back to the
// hardcoded package default if containers.conf can't be loaded at all.
func resolvedCDISpecDirs() []string {
	cfg, err := config.Default()
	if err != nil {
		logrus.Warnf("Failed to load containers.conf: %v", err)
		return config.DefaultCdiSpecDirs
	}
	return cfg.Engine.CdiSpecDirs.Get()
}

// injectCDISpecDirs prepends one --cdi-spec-dir=<dir> occurrence per entry
// in dirs to os.Args, before cobra ever parses it. This is safe
// specifically because it runs from an init(): every package's init()
// (this one included) is guaranteed by the Go spec to finish before
// main() starts, and main() is what calls rootCmd.Execute(), which reads
// os.Args itself (nothing in this tree calls cobra's SetArgs to override
// that).
//
// --cdi-spec-dir is a StringArray flag: pflag only *appends* starting
// from the second CLI occurrence -- the first occurrence replaces the
// flag's Go-code default outright
// (vendor/github.com/spf13/pflag/string_array.go, Set()). So injecting a
// single --cdi-spec-dir=<job dir> would silently drop every other
// configured CDI spec dir for this invocation. Callers must pass the
// full list they want in effect (see resolvedCDISpecDirs), including the
// job dir, so this reconstructs the set instead of clobbering it.
func injectCDISpecDirs(dirs []string) {
	extra := make([]string, 0, len(dirs))
	for _, d := range dirs {
		extra = append(extra, "--cdi-spec-dir="+d)
	}
	args := make([]string, 0, len(os.Args)+len(extra))
	args = append(args, os.Args[0])
	args = append(args, extra...)
	args = append(args, os.Args[1:]...)
	os.Args = args
}

// injectAnnotation inserts "--annotation <key>=<value>" right after the
// run/create subcommand token in os.Args. Unlike --cdi-spec-dir,
// --annotation is a *local* flag registered only on run/create
// (cmd/podman/common/create.go), not a persistent flag on rootCmd
// (cmd.PersistentFlags(), see root.go's pFlags), so cobra only recognizes
// it once parsing has descended into that subcommand. Placing it before
// the subcommand name like injectCDISpecDirs does would fail with
// "unknown flag: --annotation" while still parsing at the root level
// (cobra's Command.Traverse calls ParseFlags against whatever command it
// is currently at, before it has found "run"/"create").
//
// Uses createSubcommandIndex to find where run/create starts -- see that
// function's doc for why (and its stated limitations, e.g. no support
// for aliases like "container run"). Safe to call more than once per
// invocation (e.g. once for SLURM_JOB_ID, once for SLURM_CDI_DIR): each
// call re-finds the subcommand index against the current os.Args and
// inserts its own pair there.
func injectAnnotation(key, value string) {
	idx := createSubcommandIndex(os.Args)
	if idx == -1 {
		return
	}

	extra := []string{"--annotation", key + "=" + value}
	args := make([]string, 0, len(os.Args)+len(extra))
	args = append(args, os.Args[:idx+1]...)
	args = append(args, extra...)
	args = append(args, os.Args[idx+1:]...)
	os.Args = args
}

// rawNvidiaDevicePath returns the first raw host device path requested for
// this container that looks like an NVIDIA device node (e.g.
// --device=/dev/nvidia0), or "" if none was requested. Only --gpus and
// --device=nvidia.com/gpu=... (CDI) are allowed; raw device paths bypass
// CDI entirely, so nvidia-ctk never sees them and the SLURM accounting
// below has no idea the container is using a GPU.
//
// Callers must check usesNvidiaGPU first and skip this call if it's true:
// info.DevicePaths comes from the container's final OCI spec
// (runLifecycleHooks in libpod/lifecycle_hooks.go), which by BeforeStart
// already has CDI's own resolved device nodes (e.g. /dev/nvidia-modeset)
// merged into Linux.Devices alongside any genuinely raw ones. There is
// nothing left at that point to tell a CDI-injected node apart from a
// raw --device=/dev/nvidia0 by path alone, so this would otherwise flag
// legitimate --gpus/--device=nvidia.com/gpu=... requests too.
func rawNvidiaDevicePath(info *libpod.LifecycleHookInfo) string {
	for _, dev := range info.DevicePaths {
		if strings.HasPrefix(dev, "/dev/nvidia") {
			return dev
		}
	}
	return ""
}

// usesNvidiaGPU reports whether info's container actually requested an
// nvidia.com/gpu CDI device -- i.e. was created with --gpus or
// --device=nvidia.com/gpu=.... This reads back the container's own
// recorded config (info.CDIDevices), unlike wantsGPU's argv-only guess:
// a hook like AfterStop runs during "podman stop", whose os.Args never
// carries --gpus/--device at all (that command doesn't accept them), so
// there is nothing to scrape from argv at that point in the lifecycle.
func usesNvidiaGPU(info *libpod.LifecycleHookInfo) bool {
	for _, dev := range info.CDIDevices {
		if strings.HasPrefix(dev, "nvidia.com/gpu") {
			return true
		}
	}
	return false
}

func init() {
	jobID := os.Getenv("SLURM_JOB_ID")

	if jobID != "" {
		injectAnnotation("SLURM_JOB_ID", jobID)
	}

	// The CDI dir is keyed by a fresh UUID per invocation, not by jobID:
	// jobID alone would be shared by every container in the same SLURM
	// job, so the first one to stop (removeCDIDir below) would delete
	// the CDI dir out from under the others. A per-container ID avoids
	// that entirely -- each container gets, and later cleans up, its
	// own directory.
	if jobID != "" && wantsGPU(os.Args) {
		cdiDirID := uuid.NewString()
		dir := filepath.Join("/var/run/cdi", cdiDirID)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			logrus.Warnf("Failed to create CDI directory %s: %v", dir, err)
		} else {
			injectAnnotation("SLURM_CDI_DIR", cdiDirID)
			generateNvidiaCDI(dir)
			injectCDISpecDirs(append(resolvedCDISpecDirs(), dir))
		}
	}

	libpod.RegisterLifecycleHook(libpod.BeforeStart, func(ctx context.Context, info *libpod.LifecycleHookInfo) error {
		if usesNvidiaGPU(info) {
			return nil
		}
		if dev := rawNvidiaDevicePath(info); dev != "" {
			return fmt.Errorf("Raw device path %q is not allowed for NVIDIA GPUs; use --gpus or --device=nvidia.com/gpu=... instead", dev)
		}

		return nil
	})

	libpod.RegisterLifecycleHook(libpod.AfterStart, func(ctx context.Context, info *libpod.LifecycleHookInfo) error {
		return nil
	})

	libpod.RegisterLifecycleHook(libpod.BeforeStop, func(ctx context.Context, info *libpod.LifecycleHookInfo) error {
		return nil
	})

	// Registered on both AfterStop and AfterCleanup, not just AfterStop:
	// AfterStop is skipped whenever conmon's exit-command cleanup process
	// wins its race with an explicit "podman stop" (see the doc comment
	// on libpod.AfterStop) -- which for a --rm container means it gets
	// removed before AfterStop ever runs. AfterCleanup always runs, so
	// registering there too guarantees this fires at least once.
	// os.RemoveAll makes running it twice for the same container harmless.
	removeCDIDir := func(ctx context.Context, info *libpod.LifecycleHookInfo) error {
		// Reads info.Annotations, not a package-level variable: this hook
		// runs during whatever process ends up stopping/cleaning up the
		// container, which may be a different invocation (with no
		// SLURM_CDI_DIR of its own) than the one that created it.
		// info.Annotations reflects what was recorded on the container
		// itself at create time (via injectAnnotation), so this always
		// finds the right directory -- and because SLURM_CDI_DIR is a
		// fresh UUID minted per container rather than shared per
		// SLURM_JOB_ID, removing it here can never affect another
		// container's CDI spec, even one from the same job.
		if cdiDirID := info.Annotations["SLURM_CDI_DIR"]; cdiDirID != "" && usesNvidiaGPU(info) {
			dir := filepath.Join("/var/run/cdi", cdiDirID)
			if err := os.RemoveAll(dir); err != nil {
				logrus.Warnf("Failed to remove CDI directory %s: %v", dir, err)
			}
		}

		return nil
	}
	libpod.RegisterLifecycleHook(libpod.AfterStop, removeCDIDir)
	libpod.RegisterLifecycleHook(libpod.AfterCleanup, removeCDIDir)
}
