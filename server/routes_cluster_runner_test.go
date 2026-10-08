package server

import (
	"errors"
	"strings"
	"testing"

	"github.com/borism/ollama-cluster/cluster"
	"github.com/borism/ollama-cluster/manifest"
	"github.com/borism/ollama-cluster/types/model"
)

func TestResolveModelClusterRunnerPreference(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	// Simulate Apple Silicon's default (mlx first).
	old := manifest.DefaultRunnerPreferences
	manifest.DefaultRunnerPreferences = func() []string {
		return []string{manifest.RunnerMLX, manifest.RunnerLlamaCPP, manifest.RunnerGGML}
	}
	t.Cleanup(func() { manifest.DefaultRunnerPreferences = old })

	writeChild := func(name, runner, format string) {
		t.Helper()
		config := makeManifestListConfig(t, format)
		layer, err := manifest.NewLayer(strings.NewReader(name+" weights"), "application/vnd.ollama.image.model")
		if err != nil {
			t.Fatal(err)
		}
		if err := manifest.WriteManifestWithMetadata(model.ParseName(name), config, []manifest.Layer{layer}, runner, format); err != nil {
			t.Fatal(err)
		}
	}
	writeChild("library/both-mlx:latest", manifest.RunnerMLX, manifest.FormatSafetensors)
	writeChild("library/both-llama:latest", manifest.RunnerLlamaCPP, manifest.FormatGGUF)
	writeManifestListFixture(t, "library/both:latest",
		manifestListFixtureChild{"library/both-mlx:latest", manifest.RunnerMLX, manifest.FormatSafetensors},
		manifestListFixtureChild{"library/both-llama:latest", manifest.RunnerLlamaCPP, manifest.FormatGGUF})
	writeChild("library/onlymlx-mlx:latest", manifest.RunnerMLX, manifest.FormatSafetensors)
	writeManifestListFixture(t, "library/onlymlx:latest",
		manifestListFixtureChild{"library/onlymlx-mlx:latest", manifest.RunnerMLX, manifest.FormatSafetensors})

	oldDial := dialRPC
	t.Cleanup(func() { dialRPC = oldDial })

	s := &Server{sched: &Scheduler{}}
	s.sched.clusterTable.Store(cluster.NewStaticTable(cluster.Peer{ID: "p", Addr: "192.0.2.1", RPCPort: 50052}))
	up := func(string) error { return nil }
	down := func(string) error { return errors.New("down") }
	off := false

	for _, tc := range []struct {
		name, model, runner string
		opts                map[string]any
		dial                func(string) error
		want                string
	}{
		{"peer reachable", "library/both:latest", "", nil, up, manifest.RunnerLlamaCPP},
		{"no peer reachable", "library/both:latest", "", nil, down, manifest.RunnerMLX},
		{"explicit runner wins", "library/both:latest", manifest.RunnerMLX, nil, up, manifest.RunnerMLX},
		{"rpc_auto false", "library/both:latest", "", map[string]any{"rpc_auto": off}, up, manifest.RunnerMLX},
		{"only mlx child", "library/onlymlx:latest", "", nil, up, manifest.RunnerMLX},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dialRPC = tc.dial
			m, err := s.resolveModel(tc.model, tc.runner, tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			if m.Runner != tc.want {
				t.Fatalf("runner = %q, want %q", m.Runner, tc.want)
			}
		})
	}

	t.Run("cluster off", func(t *testing.T) {
		s.sched.clusterTable.Store(nil)
		dialRPC = up
		m, err := s.resolveModel("library/both:latest", "", nil)
		if err != nil || m.Runner != manifest.RunnerMLX {
			t.Fatalf("runner = %v, err = %v, want mlx", m, err)
		}
	})
}
