package envconfig

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// The cluster settings below each come from, in order: their OLLAMA_CLUSTER*
// environment variable, the matching key in ~/.ollama/server.json (what
// `ollama cluster on/off/set` writes, through the server), or the default.
// Reading server.json too means a service whose environment is fixed (a
// systemd unit on Linux) can be switched without editing it.

// Cluster turns on LAN autodiscovery of other ollama-cluster instances
// (see cluster.Start) and, unless ClusterShare is set false, donating this
// instance's spare GPU capacity as an RPC worker for them
// (llm.StartRPCWorker). Opt-in: llama.cpp's RPC backend is documented
// insecure upstream (tools/rpc/README.md in ggml-org/llama.cpp), this
// assumes a trusted LAN.
func Cluster() bool {
	return clusterBool("OLLAMA_CLUSTER", serverConfigValue(func(c serverConfigData) *bool { return c.Cluster }), false)
}

// ClusterShare controls whether this instance advertises spare GPU
// capacity to the cluster; only meaningful when Cluster is on. Set it
// false to consume peers' capacity without donating this instance's own.
func ClusterShare(defaultValue bool) bool {
	return clusterBool("OLLAMA_CLUSTER_SHARE", serverConfigValue(func(c serverConfigData) *bool { return c.ClusterShare }), defaultValue)
}

// ClusterShareDevices is a comma-separated list of ggml device names
// ("CUDA1,Vulkan0") to share instead of every accelerator, passed to the
// RPC worker as -d (tools/rpc/rpc-server.cpp in ggml-org/llama.cpp) and used
// to filter what discovery advertises. Empty means all devices.
func ClusterShareDevices() []string {
	return splitList(clusterString("OLLAMA_CLUSTER_SHARE_DEVICES", serverConfigValue(func(c serverConfigData) *string { return c.ClusterShareDevices }), ""))
}

// ClusterSeeds is a comma-separated "host:port,host:port" list of
// cluster-discovery beacon addresses to unicast directly to, for reaching
// peers outside this host's broadcast domain (see cluster.Config.Seeds --
// UDP broadcast doesn't cross a subnet/VLAN). Only meaningful when Cluster
// is on. Empty by default: same-subnet broadcast discovery needs no seeds.
func ClusterSeeds() []string {
	return splitList(clusterString("OLLAMA_CLUSTER_SEEDS", serverConfigValue(func(c serverConfigData) *string { return c.ClusterSeeds }), ""))
}

// splitList splits a comma list, trimming entries and dropping empty ones.
func splitList(raw string) []string {
	var out []string
	for _, s := range strings.Split(raw, ",") {
		if s = strings.TrimSpace(s); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// ClusterPlacement selects how cluster.SelectRPCServers picks peers: "" or
// "waterfill" (default) weighs peers by load/latency and can spread a
// shortfall across several of them; "greedy" restores the original v1
// behavior -- peers ranked by free memory, added until the shortfall is
// covered, ignoring load/latency. Overridable per request via
// api.Options.RPCPlacement.
func ClusterPlacement() string {
	return clusterString("OLLAMA_CLUSTER_PLACEMENT", serverConfigValue(func(c serverConfigData) *string { return c.ClusterPlacement }), "")
}

// ClusterCacheGB caps the shared RPC worker's tensor cache (weights peers
// loaded onto this machine, kept so a reload skips the network; see
// llm/rpc_cache.go). 0 turns the cache off.
func ClusterCacheGB() uint {
	if s := Var("OLLAMA_CLUSTER_CACHE_GB"); s != "" {
		if n, err := strconv.ParseUint(s, 10, 32); err == nil {
			return uint(n)
		}
	}
	if v := serverConfigValue(func(c serverConfigData) *uint { return c.ClusterCacheGB }); v != nil {
		return *v
	}
	return 32
}

// ClusterSources reports where each cluster setting's value comes from:
// "env", "config" (server.json) or "default". Keys are the
// api.ClusterConfig JSON names.
func ClusterSources() map[string]string {
	loadServerConfig()
	c := cachedServerConfig()
	source := func(key string, inConfig bool) string {
		switch {
		case Var(key) != "":
			return "env"
		case inConfig:
			return "config"
		default:
			return "default"
		}
	}
	return map[string]string{
		"enabled":       source("OLLAMA_CLUSTER", c.Cluster != nil),
		"share":         source("OLLAMA_CLUSTER_SHARE", c.ClusterShare != nil),
		"seeds":         source("OLLAMA_CLUSTER_SEEDS", c.ClusterSeeds != nil),
		"share_devices": source("OLLAMA_CLUSTER_SHARE_DEVICES", c.ClusterShareDevices != nil),
		"placement":     source("OLLAMA_CLUSTER_PLACEMENT", c.ClusterPlacement != nil),
		"cache_gb":      source("OLLAMA_CLUSTER_CACHE_GB", c.ClusterCacheGB != nil),
	}
}

// UpdateServerConfig sets the given top-level keys in ~/.ollama/server.json
// (a nil value removes the key), keeps every other key as it is, and
// reloads the cached settings. The file is replaced atomically.
func UpdateServerConfig(set map[string]any) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	path := filepath.Join(home, ".ollama", "server.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create server config directory: %w", err)
	}

	cfg := map[string]any{}
	if data, err := os.ReadFile(path); err == nil {
		if err := json.Unmarshal(data, &cfg); err != nil {
			return fmt.Errorf("%s is not valid JSON, not overwriting it: %w", path, err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("read server config: %w", err)
	}
	for k, v := range set {
		if v == nil {
			delete(cfg, k)
		} else {
			cfg[k] = v
		}
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), "server.json.*")
	if err != nil {
		return fmt.Errorf("write server config: %w", err)
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		tmp.Close()
		return fmt.Errorf("write server config: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("write server config: %w", err)
	}
	if err := os.Chmod(tmp.Name(), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return fmt.Errorf("write server config: %w", err)
	}
	ReloadServerConfig()
	return nil
}

func serverConfigValue[T any](field func(serverConfigData) *T) *T {
	loadServerConfig()
	return field(cachedServerConfig())
}

func clusterBool(key string, config *bool, defaultValue bool) bool {
	if Var(key) != "" {
		return BoolWithDefault(key)(defaultValue)
	}
	if config != nil {
		return *config
	}
	return defaultValue
}

func clusterString(key string, config *string, defaultValue string) string {
	if s := Var(key); s != "" {
		return s
	}
	if config != nil {
		return *config
	}
	return defaultValue
}
