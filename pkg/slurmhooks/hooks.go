//go:build !remote

// Package slurmhooks wires podman's container lifecycle into SLURM.
//
// It is registered as a blank import from cmd/podman/main.go, so it only
// needs to be linked in once; from then on every "podman run/start/stop"
// on this host runs it automatically, in-process, for every container.
//
// GPU device isolation depends on containers ending up *inside* the
// cgroup SLURM created for the job, so cgroup.conf's ConstrainDevices=yes
// (and any CPU/memory limits on the job) apply to them. init() below
// injects --cgroup-manager=cgroupfs, --cgroup-parent=<task cgroup> and
// --cgroupns=host into every GPU-requesting invocation to make that
// happen automatically. That in turn requires the task's cgroup
// (normally root:root) to be writable by the job's own uid, which this
// package cannot do on its own -- it depends on a companion SPANK
// plugin (tmp/spank/cgroup_delegate.c in the podman tree) being deployed
// on the SLURM cluster to chown() it at task launch time. Without that
// plugin, --cgroup-parent will fail with a permission error rather than
// silently doing nothing -- see injectCgroupParent's doc for why.
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

// injectLocalFlags inserts each of tokens, in order, right after the
// run/create subcommand token in os.Args. --annotation, --cgroup-parent
// and --cgroupns are all *local* flags registered only on run/create
// (cmd/podman/common/create.go), not persistent flags on rootCmd
// (cmd.PersistentFlags(), see root.go's pFlags), so cobra only recognizes
// them once parsing has descended into that subcommand. Placing them
// before the subcommand name like injectCDISpecDirs/injectCgroupManager
// do for persistent flags would fail with "unknown flag: ..." while
// still parsing at the root level (cobra's Command.Traverse calls
// ParseFlags against whatever command it is currently at, before it has
// found "run"/"create").
//
// Uses createSubcommandIndex to find where run/create starts -- see that
// function's doc for why (and its stated limitations, e.g. no support
// for aliases like "container run"). Safe to call more than once per
// invocation: each call re-finds the subcommand index against the
// current os.Args and inserts its own tokens there.
func injectLocalFlags(tokens ...string) {
	idx := createSubcommandIndex(os.Args)
	if idx == -1 {
		return
	}

	args := make([]string, 0, len(os.Args)+len(tokens))
	args = append(args, os.Args[:idx+1]...)
	args = append(args, tokens...)
	args = append(args, os.Args[idx+1:]...)
	os.Args = args
}

// injectAnnotation inserts "--annotation <key>=<value>" right after the
// run/create subcommand token in os.Args (via injectLocalFlags). Safe to
// call more than once per invocation (e.g. once for SLURM_JOB_ID, once
// for SLURM_CDI_DIR).
func injectAnnotation(key, value string) {
	injectLocalFlags("--annotation", key+"="+value)
}

// ownCgroupPath returns this process's own cgroup path from the unified
// (cgroup v2) hierarchy line in /proc/self/cgroup, e.g.
// "/system.slice/slurmstepd.scope/job_128634/step_0/user/task_0" for a
// process SLURM placed under a job step's task cgroup.
//
// Safe to read directly here (unlike the companion cgroup_delegate SPANK
// plugin, which has to reconstruct this same path from SLURM job/step/
// task ids instead of trusting its own /proc/self/cgroup): this code
// runs in the podman *client* process, which is the shell SLURM already
// placed directly into the task's cgroup. cgroup_delegate runs deep
// inside slurmstepd itself, in a sibling "slurm" bookkeeping cgroup that
// is never the task's own -- see that plugin's source for the story of
// how that was confirmed by testing.
func ownCgroupPath() (string, error) {
	data, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(string(data), "\n") {
		if path, ok := strings.CutPrefix(line, "0::"); ok {
			return strings.TrimSpace(path), nil
		}
	}
	return "", fmt.Errorf("no cgroup v2 (0::) entry in /proc/self/cgroup")
}

// injectCgroupManager prepends "--cgroup-manager=<manager>" to os.Args,
// before the run/create subcommand token -- --cgroup-manager is a
// persistent flag on rootCmd (cmd/podman/root.go), unlike --cgroup-parent
// or --cgroupns, so it must come before the subcommand the same way
// injectCDISpecDirs's --cdi-spec-dir does (see that function's doc for
// why cobra requires this ordering). Required for --cgroup-parent below:
// podman's default systemd cgroup manager expects a slice/scope-style
// name, not an arbitrary raw cgroupfs path like SLURM's.
func injectCgroupManager(manager string) {
	args := make([]string, 0, len(os.Args)+1)
	args = append(args, os.Args[0])
	args = append(args, "--cgroup-manager="+manager)
	args = append(args, os.Args[1:]...)
	os.Args = args
}

// injectCgroupParent inserts "--cgroup-parent=<path>" and
// "--cgroupns=host" right after the run/create subcommand token (via
// injectLocalFlags). Forces the new container's cgroup to be created as
// a descendant of path -- SLURM's own cgroup for this job's task --
// instead of wherever podman's cgroup-manager would otherwise place it
// (rootless podman's usual user.slice/user@<uid>.service/.../<id>.scope,
// a sibling of SLURM's cgroup, not a child of it). --cgroupns=host makes
// the container share the host's cgroup namespace instead of getting its
// own, so it's actually visible as nested there (with a private
// namespace, /proc/self/cgroup inside the container would just show "/"
// regardless of where its cgroup really lives).
//
// This matters because a cgroup v2 device-access eBPF program
// (cgroup.conf's ConstrainDevices=yes) and any CPU/memory limits SLURM
// placed on the job's cgroup only apply to that cgroup and its
// descendants. A container whose cgroup isn't nested under the job's is
// invisible to those constraints and can open any GPU device node on the
// host and use unlimited CPU/memory, regardless of what SLURM actually
// allocated to the job.
//
// Requires path to already be writable by the current (unprivileged)
// user -- by default SLURM's job/step/task cgroups are root:root all the
// way down, so podman creating a child cgroup under one fails with a
// plain permission error. Getting that delegation is exactly what the
// companion cgroup_delegate SPANK plugin (tmp/spank/cgroup_delegate.c)
// does; without it deployed on the cluster, this will fail loudly
// (crun: create `.../libpod-<id>`: Permission denied) rather than
// silently leaving the container unconfined.
func injectCgroupParent(path string) {
	injectLocalFlags("--cgroup-parent="+path, "--cgroupns=host")
}

// rawNvidiaDevicePath returns the first raw host device path requested for
// this container that looks like an NVIDIA device node (e.g.
// --device=/dev/nvidia0), or "" if none was requested. Only --gpus and
// --device=nvidia.com/gpu=... (CDI) are allowed.
//
// This is not the GPU isolation boundary -- that's injectCgroupParent
// above, which puts every GPU-requesting container inside the SLURM
// job's own cgroup so cgroup.conf's ConstrainDevices=yes applies to it
// regardless of how the device was requested, raw path or CDI. What this
// check actually buys:
//   - raw paths skip nvidia-ctk entirely, so the container gets the
//     device node but not the matching NVIDIA userspace driver libraries
//     (libcuda.so etc.) CDI would have bind-mounted in -- the container
//     would see the GPU but couldn't actually drive it;
//   - raw paths never touch generateNvidiaCDI/injectAnnotation, so the
//     SLURM_CDI_DIR accounting below never fires for them;
//   - it's a second, independent check that doesn't depend on the
//     cgroup-parent machinery working correctly on a given node.
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
		// See injectCgroupParent's doc: nests the container under
		// SLURM's own cgroup for this task so cgroup.conf's
		// ConstrainDevices=yes (and the job's CPU/memory limits)
		// actually apply to it. Requires the companion cgroup_delegate
		// SPANK plugin to be deployed -- without it this fails loudly
		// with a permission error rather than silently skipping.
		if parent, err := ownCgroupPath(); err != nil {
			logrus.Warnf("Failed to read own cgroup path: %v", err)
		} else {
			injectCgroupManager("cgroupfs")
			injectCgroupParent(parent)
		}

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
