//go:build !remote

// Package slurmhooks nests every container podman creates into the cgroup
// SLURM created for the current job/task, so cgroup.conf's
// ConstrainDevices=yes (and any CPU/memory limits on the job) apply to it.
//
// This is applied unconditionally to every "podman run/create" under a
// SLURM job -- not just ones that look like they're requesting a GPU.
// Trying to detect "this invocation wants a GPU" from argv or env (e.g. by
// looking for --gpus/--device, or NVIDIA_VISIBLE_DEVICES) is only ever a
// heuristic: anything it fails to recognize -- a legacy NVIDIA OCI
// prestart hook keyed off an env var, a hand-rolled CDI spec, a future
// flag this package doesn't know about -- would start life outside
// SLURM's cgroup and be invisible to ConstrainDevices=yes regardless of
// how it got its device access. Gating only on "is this a container/pod
// creation at all" (isContainerCreate) removes that entire class of
// bypass: enforcement happens once, at the kernel cgroup level, for every
// container, independent of what it asked for or how.
//
// It is registered as a blank import from cmd/podman/main.go, so it only
// needs to be linked in once; from then on every "podman run/create" on
// this host runs it automatically, in-process, via init().
//
// Getting cgroup-parent to work requires the task's cgroup (normally
// root:root) to be writable by the job's own uid, which this package
// cannot do on its own -- it depends on a companion SPANK plugin
// (tmp/spank/cgroup_delegate.c in the podman tree) being deployed on the
// SLURM cluster to chown() it at task launch time. Without that plugin,
// ownCgroupPath below still succeeds (it only reads /proc/self/cgroup),
// but the actual --cgroup-parent creation later fails with a permission
// error rather than silently doing nothing -- see injectCgroupParent's
// doc for why.
package slurmhooks

import (
	"fmt"
	"os"
	"strings"

	"github.com/sirupsen/logrus"
)

// createSubcommandIndex returns the index of the first bare "run" or
// "create" token in args, or -1 if there isn't one. --cgroup-parent and
// --cgroupns are only ever valid after that point (they're local flags
// registered on run/create, cmd/podman/common/create.go, not persistent
// flags on rootCmd), so injectLocalFlags's callers need to know where
// podman's own subcommand starts before they can insert them there --
// otherwise a token that merely appears earlier or later in argv (e.g. as
// an argument to `podman exec`, or a container/image name) would be
// mistaken for the real thing. init() below also uses this directly to
// decide whether the current invocation creates a container at all.
//
// This only recognizes a bare "run" or "create" token -- not aliases like
// "container run".
func createSubcommandIndex(args []string) int {
	for i, a := range args {
		if a == "run" || a == "create" {
			return i
		}
	}
	return -1
}

// injectLocalFlags inserts each of tokens, in order, right after the
// run/create subcommand token in os.Args. --annotation, --cgroup-parent
// and --cgroupns are all *local* flags registered only on run/create
// (cmd/podman/common/create.go), not persistent flags on rootCmd
// (cmd.PersistentFlags(), see root.go's pFlags), so cobra only recognizes
// them once parsing has descended into that subcommand. Placing them
// before the subcommand name like injectCgroupManager does for persistent
// flags would fail with "unknown flag: ..." while still parsing at the
// root level (cobra's Command.Traverse calls ParseFlags against whatever
// command it is currently at, before it has found "run"/"create").
//
// Uses createSubcommandIndex to find where run/create starts -- see that
// function's doc for why (and its stated limitations, e.g. no support for
// aliases like "container run"). Safe to call more than once per
// invocation: each call re-finds the subcommand index against the current
// os.Args and inserts its own tokens there.
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

// nonContainerCreateParents lists subcommand groups that register their
// own "create" leaf without it creating a container or pod: "podman
// <group> create". createSubcommandIndex only looks for the bare
// "create"/"run" token itself, with no awareness of which command group
// it belongs to, so isContainerCreate below needs this to tell "podman
// network create" apart from "podman pod create" or a bare "podman
// create". "system connection create" is an alias for "connection add"
// (cmd/podman/system/connection/add.go), hence "connection" rather than
// "system" here -- the immediately preceding token is what matters.
var nonContainerCreateParents = map[string]bool{
	"network":    true,
	"volume":     true,
	"secret":     true,
	"manifest":   true,
	"farm":       true,
	"connection": true,
}

// isContainerCreate reports whether idx (as returned by
// createSubcommandIndex) marks an invocation that actually creates a
// container or pod, as opposed to some other "podman <group> create"
// command that happens to share the same leaf name -- see
// nonContainerCreateParents. This is the gate init() uses to decide
// whether an invocation needs SLURM cgroup nesting at all; getting it
// wrong in the permissive direction would mean, e.g., "podman network
// create" or "podman volume create" failing outright under a SLURM job
// with "unknown flag: --cgroup-parent" the moment injectCgroupParent
// tried to inject a flag those commands don't have -- exactly the
// podman-compose failure mode this was written to catch, since compose
// creates a network and often a volume before it ever creates a
// container.
func isContainerCreate(args []string, idx int) bool {
	if idx == -1 {
		return false
	}
	return idx == 0 || !nonContainerCreateParents[args[idx-1]]
}

// isPodCreate reports whether idx (as returned by createSubcommandIndex)
// points at the "create" in "podman pod create ..." rather than a bare
// "podman create"/"podman run". Callers should only call this once
// isContainerCreate has already confirmed idx marks a real container/pod
// creation. The distinction matters because "podman pod create" shares
// its flag-registration code with "create"
// (cmd/podman/common/create.go's DefineCreateFlags), but most of that
// function's flags -- including --annotation and --cgroupns -- are gated
// on entities.CreateMode, which "pod create" (entities.InfraMode) doesn't
// use. Inserting a flag pod create doesn't have fails the whole command
// with "unknown flag: ...", so callers that would otherwise inject one of
// those need to special-case this. podman-compose, and anything else that
// shares a pod across multiple containers, always calls "podman pod
// create" first, so this isn't just a concern for direct pod-create use.
func isPodCreate(args []string, idx int) bool {
	return idx > 0 && args[idx-1] == "pod"
}

// injectAnnotation inserts "--annotation <key>=<value>" right after the
// run/create subcommand token in os.Args (via injectLocalFlags). Safe to
// call more than once per invocation.
//
// No-ops for anything isContainerCreate doesn't recognize as a real
// container/pod creation (e.g. "podman network create", which has no
// --annotation flag either) and for "pod create" specifically (see
// isPodCreate) since that command has no --annotation flag at all. The
// pod's own member containers (created via separate "podman create
// --pod=..." invocations, which do carry --annotation) still get it; only
// the pod object, its infra container, and non-container "create" leaves
// don't.
func injectAnnotation(key, value string) {
	idx := createSubcommandIndex(os.Args)
	if !isContainerCreate(os.Args, idx) || isPodCreate(os.Args, idx) {
		return
	}
	injectLocalFlags("--annotation", key+"="+value)
}

// ownCgroupPath returns this process's own cgroup path from the unified
// (cgroup v2) hierarchy line in /proc/self/cgroup, e.g.
// "/system.slice/slurmstepd.scope/job_128634/step_0/user/task_0" for a
// process SLURM placed under a job step's task cgroup.
//
// Safe to read directly here (unlike the companion cgroup_delegate SPANK
// plugin, which has to reconstruct this same path from SLURM job/step/
// task ids instead of trusting its own /proc/self/cgroup): this code runs
// in the podman *client* process, which is the shell SLURM already placed
// directly into the task's cgroup. cgroup_delegate runs deep inside
// slurmstepd itself, in a sibling "slurm" bookkeeping cgroup that is never
// the task's own -- see that plugin's source for the story of how that
// was confirmed by testing.
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
// or --cgroupns, so it must come before the subcommand the same way a
// persistent flag would (cobra parses persistent flags at whatever level
// of the command tree it's currently traversing, before it has descended
// into "run"/"create"). Required for --cgroup-parent below: podman's
// default systemd cgroup manager expects a slice/scope-style name, not an
// arbitrary raw cgroupfs path like SLURM's.
func injectCgroupManager(manager string) {
	args := make([]string, 0, len(os.Args)+1)
	args = append(args, os.Args[0])
	args = append(args, "--cgroup-manager="+manager)
	args = append(args, os.Args[1:]...)
	os.Args = args
}

// injectCgroupParent inserts "--cgroup-parent=<path>" right after the
// run/create subcommand token (via injectLocalFlags), plus "--cgroupns=host"
// for a plain "run"/"create" -- but not for "pod create": see isPodCreate.
// --cgroup-parent is registered for both "create"/"run" and "pod create"
// (unlike --annotation/--cgroupns, which "pod create" lacks entirely) and
// is all the pod itself needs: its infra container's CgroupParent is
// forced to the pod's own regardless of any --cgroupns setting
// (pkg/specgen/generate/pod_create.go), and the infra container isn't
// built from a second parsed CLI invocation this package could intercept
// anyway.
//
// --cgroup-parent forces the new container's (or pod's) cgroup to be
// created as a descendant of path -- SLURM's own cgroup for this job's
// task -- instead of wherever podman's cgroup-manager would otherwise
// place it (rootless podman's usual
// user.slice/user@<uid>.service/.../<id>.scope, a sibling of SLURM's
// cgroup, not a child of it). --cgroupns=host makes a "run"/"create"
// container share the host's cgroup namespace instead of getting its own,
// so it's actually visible as nested there (with a private namespace,
// /proc/self/cgroup inside the container would just show "/" regardless
// of where its cgroup really lives).
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
	if isPodCreate(os.Args, createSubcommandIndex(os.Args)) {
		injectLocalFlags("--cgroup-parent=" + path)
		return
	}
	injectLocalFlags("--cgroup-parent="+path, "--cgroupns=host")
}

func init() {
	jobID := os.Getenv("SLURM_JOB_ID")

	if jobID != "" {
		injectAnnotation("SLURM_JOB_ID", jobID)
	}

	// Applied to every container/pod creation under a SLURM job, not
	// just ones that look like they're requesting a GPU -- see the
	// package doc for why this isn't gated on trying to guess intent
	// from argv or env. isContainerCreate (as opposed to a bare
	// createSubcommandIndex(os.Args) != -1 check) is what keeps this
	// from also firing on "podman network create", "podman volume
	// create", and the other non-container commands that happen to
	// share the "create" leaf name.
	if jobID != "" && isContainerCreate(os.Args, createSubcommandIndex(os.Args)) {
		parent, err := ownCgroupPath()
		if err != nil {
			// Fail closed: if we can't determine where SLURM's cgroup
			// actually is, we cannot nest the container under it, and
			// letting the container start anyway would run it fully
			// unconfined (no device restriction, no CPU/memory limit)
			// with no indication anything went wrong. Refuse to create
			// the container at all instead.
			logrus.Fatalf("slurmhooks: failed to read own cgroup path, refusing to start container unconfined: %v", err)
		}
		injectCgroupManager("cgroupfs")
		injectCgroupParent(parent)
	}
}
