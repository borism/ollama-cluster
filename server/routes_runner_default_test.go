package server

import (
	"strings"
	"testing"

	"github.com/borism/ollama-cluster/manifest"
	"github.com/borism/ollama-cluster/types/model"
)

// This fork prefers cluster-capable (llama.cpp) builds on every platform; only
// an explicit runner or an MLX-only list yields MLX.
func TestGetModelPrefersLlamaCPPOverMLX(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	writeChild := func(name, runner, format string) {
		t.Helper()
		layer, err := manifest.NewLayer(strings.NewReader(name+" weights"), "application/vnd.ollama.image.model")
		if err != nil {
			t.Fatal(err)
		}
		if err := manifest.WriteManifestWithMetadata(model.ParseName(name), makeManifestListConfig(t, format), []manifest.Layer{layer}, runner, format); err != nil {
			t.Fatal(err)
		}
	}
	writeChild("library/both-mlx:latest", manifest.RunnerMLX, manifest.FormatSafetensors)
	writeChild("library/both-llama:latest", manifest.RunnerLlamaCPP, manifest.FormatGGUF)
	mlx := manifestListFixtureChild{"library/both-mlx:latest", manifest.RunnerMLX, manifest.FormatSafetensors}
	writeManifestListFixture(t, "library/both:latest", mlx, manifestListFixtureChild{"library/both-llama:latest", manifest.RunnerLlamaCPP, manifest.FormatGGUF})
	writeManifestListFixture(t, "library/onlymlx:latest", mlx)

	for _, tc := range []struct{ name, model, runner, want string }{
		{"default", "library/both:latest", "", manifest.RunnerLlamaCPP},
		{"explicit mlx", "library/both:latest", manifest.RunnerMLX, manifest.RunnerMLX},
		{"mlx only", "library/onlymlx:latest", "", manifest.RunnerMLX},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, err := GetModelForRunner(tc.model, tc.runner)
			if err != nil {
				t.Fatal(err)
			}
			if m.Runner != tc.want {
				t.Fatalf("runner = %q, want %q", m.Runner, tc.want)
			}
		})
	}
}
