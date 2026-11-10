package mlxrunner

import (
	"testing"

	"github.com/borism/ollama-cluster/mlx"
)

func TestSupportsRippleArchitecture(t *testing.T) {
	if !SupportsArchitecture("RicursiveForCausalLM") {
		t.Fatal("missing model architecture")
	}
}

func TestCheckRuntimeRequiresLoadedLibrary(t *testing.T) {
	_, loadErr := mlx.LoadedLibraryPath()
	err := CheckRuntime()
	if (err == nil) != (loadErr == nil) {
		t.Fatalf("CheckRuntime() error = %v, loaded library error = %v", err, loadErr)
	}
}
