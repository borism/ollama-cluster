package envconfig

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestClusterSettingsPrecedence: an OLLAMA_CLUSTER* variable beats
// server.json, which beats the default.
func TestClusterSettingsPrecedence(t *testing.T) {
	setTestHome(t, t.TempDir())
	for _, k := range []string{"OLLAMA_CLUSTER", "OLLAMA_CLUSTER_SHARE", "OLLAMA_CLUSTER_SEEDS", "OLLAMA_CLUSTER_SHARE_DEVICES", "OLLAMA_CLUSTER_PLACEMENT", "OLLAMA_CLUSTER_CACHE_GB"} {
		t.Setenv(k, "")
	}

	if Cluster() || !ClusterShare(true) || ClusterCacheGB() != 32 || ClusterSources()["enabled"] != "default" {
		t.Fatalf("defaults: cluster=%v share=%v cache=%d source=%q", Cluster(), ClusterShare(true), ClusterCacheGB(), ClusterSources()["enabled"])
	}

	if err := UpdateServerConfig(map[string]any{
		"cluster":               true,
		"cluster_share":         false,
		"cluster_seeds":         "192.0.2.10:11435",
		"cluster_share_devices": " CUDA1, Vulkan0 ,",
		"cluster_placement":     "greedy",
		"cluster_cache_gb":      8,
	}); err != nil {
		t.Fatal(err)
	}
	if !Cluster() || ClusterShare(true) || ClusterPlacement() != "greedy" || ClusterCacheGB() != 8 ||
		len(ClusterSeeds()) != 1 || ClusterSources()["share"] != "config" {
		t.Errorf("from server.json: cluster=%v share=%v placement=%q cache=%d seeds=%v sources=%v",
			Cluster(), ClusterShare(true), ClusterPlacement(), ClusterCacheGB(), ClusterSeeds(), ClusterSources())
	}

	if d := ClusterShareDevices(); len(d) != 2 || d[0] != "CUDA1" || d[1] != "Vulkan0" || ClusterSources()["share_devices"] != "config" {
		t.Errorf("share devices from server.json: %q sources=%v", d, ClusterSources())
	}
	t.Setenv("OLLAMA_CLUSTER_SHARE_DEVICES", "CUDA0")
	if d := ClusterShareDevices(); len(d) != 1 || d[0] != "CUDA0" || ClusterSources()["share_devices"] != "env" {
		t.Errorf("share devices env should win: %q", d)
	}

	t.Setenv("OLLAMA_CLUSTER", "0")
	t.Setenv("OLLAMA_CLUSTER_CACHE_GB", "64")
	if Cluster() || ClusterCacheGB() != 64 || ClusterSources()["enabled"] != "env" {
		t.Errorf("env should win: cluster=%v cache=%d source=%q", Cluster(), ClusterCacheGB(), ClusterSources()["enabled"])
	}
}

// TestUpdateServerConfigKeepsOtherKeys: writing cluster keys must not drop
// what the desktop app keeps in the same file (disable_ollama_cloud), and a
// nil value removes a key.
func TestUpdateServerConfigKeepsOtherKeys(t *testing.T) {
	home := t.TempDir()
	setTestHome(t, home)
	path := filepath.Join(home, ".ollama", "server.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"disable_ollama_cloud": true, "cluster_seeds": "192.0.2.10:11435"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := UpdateServerConfig(map[string]any{"cluster": true, "cluster_seeds": nil}); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got["disable_ollama_cloud"] != true || got["cluster"] != true {
		t.Errorf("server.json = %s, want disable_ollama_cloud and cluster both true", data)
	}
	if _, ok := got["cluster_seeds"]; ok {
		t.Errorf("server.json = %s, want cluster_seeds removed", data)
	}
	if !NoCloud() {
		t.Error("NoCloud() = false after a cluster update, want the cloud setting kept")
	}
}
