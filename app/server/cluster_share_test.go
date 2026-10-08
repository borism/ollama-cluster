package server

import (
	"testing"

	"github.com/borism/ollama-cluster/api"
)

func TestClusterShare(t *testing.T) {
	two := []api.ClusterDevice{{Name: "CUDA0", Description: "A"}, {Name: "CUDA1", Description: "B"}}
	same := []api.ClusterDevice{{Name: "CUDA0", Description: "A"}, {Name: "CUDA1", Description: "A"}}
	cfg := func(share bool, sd string, d []api.ClusterDevice) api.ClusterConfig {
		return api.ClusterConfig{Share: share, ShareDevices: sd, Devices: d}
	}
	for _, tt := range []struct {
		name  string
		c     api.ClusterConfig
		dev   string
		share bool
		sd    string
	}{
		{"all on, flip one off", cfg(true, "", two), "CUDA1", true, "CUDA0"},
		{"one on, flip other on = all", cfg(true, "cuda0", two), "CUDA1", true, ""},
		{"last off", cfg(true, "CUDA0", two), "CUDA0", false, ""},
		{"sharing off, flip one on", cfg(false, "", two), "CUDA1", true, "CUDA1"},
		{"single gpu off", cfg(true, "", two[:1]), "CUDA0", false, ""},
		{"cpu only", cfg(false, "", nil), "", true, ""},
	} {
		share, sd := ClusterShareToggle(tt.c, tt.dev)
		if share != tt.share || sd != tt.sd {
			t.Errorf("%s: got %v %q, want %v %q", tt.name, share, sd, tt.share, tt.sd)
		}
	}

	for _, tt := range []struct {
		c    api.ClusterConfig
		want string
	}{
		{cfg(true, "CUDA0", two), "Share A+|Share B|"},
		{cfg(true, "", same), "Share A (CUDA0)+|Share A (CUDA1)+|"},
		{cfg(true, "", two[:1]), "Share the A GPU+|"},
		{cfg(false, "", nil), "Share this computer's CPU|"},
	} {
		got := ""
		for _, it := range ClusterShareItems(tt.c) {
			got += it.Title
			if it.On {
				got += "+"
			}
			got += "|"
		}
		if got != tt.want {
			t.Errorf("items %q, want %q", got, tt.want)
		}
	}
}
