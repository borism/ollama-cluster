package manifest

import "testing"

// Pull picks its child with the same default order as load.
func TestDefaultPullSelectsLlamaCPPChild(t *testing.T) {
	mlx, err := NewManifestReference("sha256:"+repeat('1'), RunnerMLX, FormatSafetensors)
	if err != nil {
		t.Fatal(err)
	}
	llama, err := NewManifestReference("sha256:"+repeat('2'), RunnerLlamaCPP, FormatGGUF)
	if err != nil {
		t.Fatal(err)
	}
	got, err := SelectManifestReferenceForRunner([]Manifest{mlx, llama}, "")
	if err != nil || got.Runner != RunnerLlamaCPP {
		t.Fatalf("got %+v, %v; want llamacpp child", got, err)
	}
}

func repeat(c byte) string {
	b := make([]byte, 64)
	for i := range b {
		b[i] = c
	}
	return string(b)
}
