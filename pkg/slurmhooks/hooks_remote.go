//go:build remote

// podman-remote (the client-only build) talks to a remote engine over the
// API; there is no local container process, and this process's own
// /proc/self/cgroup has no relationship to whatever host the container
// actually runs on. The cgroup-nesting this package does for the local
// (!remote) build doesn't apply. This stub only exists so that the blank
// import in cmd/podman/main.go, which is shared between the podman and
// podman-remote binaries, compiles for both.
package slurmhooks
