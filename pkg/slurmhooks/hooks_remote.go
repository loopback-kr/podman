//go:build remote

// podman-remote (the client-only build) never runs a container locally, so
// there is nothing for this package to hook into: libpod.RegisterLifecycleHook
// does not even exist in a remote build. This stub only exists so that the
// blank import in cmd/podman/main.go, which is shared between the podman
// and podman-remote binaries, compiles for both.
package slurmhooks
