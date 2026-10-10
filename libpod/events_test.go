//go:build !remote

package libpod

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExpandEventsLogFilePath(t *testing.T) {
	t.Setenv("HOME", "/home/alice")
	t.Setenv("USER", "alice")

	for _, c := range []struct {
		path, expected string
	}{
		{"/var/log/podman/events.log", "/var/log/podman/events.log"},
		{"$HOME/.local/share/containers/events.log", "/home/alice/.local/share/containers/events.log"},
		{"${HOME}/events.log", "/home/alice/events.log"},
		{"/var/log/podman/$USER/events.log", "/var/log/podman/alice/events.log"},
		{"/var/log/podman/$UID/events.log", "/var/log/podman/1000/events.log"},
		{"/var/log/podman/${UID}/events.log", "/var/log/podman/1000/events.log"},
		{"/var/log/podman/$UID-$USER//events.log", "/var/log/podman/1000-alice/events.log"},
	} {
		assert.Equal(t, c.expected, expandEventsLogFilePath(c.path, 1000), c.path)
	}
}
