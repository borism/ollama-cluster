package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/envconfig"
	"github.com/borism/ollama-cluster/ml"
)

// clusterTestHome points the server's ~/.ollama/server.json at a temp dir
// and clears any OLLAMA_CLUSTER* variable, so only server.json counts.
func clusterTestHome(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	for _, k := range []string{"OLLAMA_CLUSTER", "OLLAMA_CLUSTER_SHARE", "OLLAMA_CLUSTER_SEEDS", "OLLAMA_CLUSTER_SHARE_DEVICES", "OLLAMA_CLUSTER_PLACEMENT", "OLLAMA_CLUSTER_CACHE_GB"} {
		t.Setenv(k, "")
	}
	envconfig.ReloadServerConfig()
	t.Cleanup(envconfig.ReloadServerConfig)

	// No real GPU discovery in tests, and no list cached by an earlier test.
	orig := localGPUDevices
	deviceCache.mu.Lock()
	deviceCache.have, deviceCache.refreshing = false, false
	deviceCache.mu.Unlock()
	localGPUDevices = func(context.Context) []ml.DeviceInfo {
		return []ml.DeviceInfo{{Name: "CUDA0", Description: "Test GPU", TotalMemory: 8 << 30}}
	}
	t.Cleanup(func() { localGPUDevices = orig })
}

// GET /api/cluster/config must not re-run GPU discovery per request: it can
// take over a second, and the macOS menu bar gives the request 500 ms.
func TestClusterConfigCachesDevices(t *testing.T) {
	clusterTestHome(t)
	var calls atomic.Int32
	localGPUDevices = func(context.Context) []ml.DeviceInfo {
		calls.Add(1)
		time.Sleep(300 * time.Millisecond)
		return []ml.DeviceInfo{{Name: "CUDA0", Description: "Test GPU", TotalMemory: 8 << 30}}
	}
	origTTL := deviceCacheTTL
	deviceCacheTTL = 200 * time.Millisecond
	t.Cleanup(func() { deviceCacheTTL = origTTL })

	get := func() (api.ClusterConfig, time.Duration) {
		start := time.Now()
		cfg := currentClusterConfig(context.Background())
		return cfg, time.Since(start)
	}
	fast := 100 * time.Millisecond

	if cfg, _ := get(); len(cfg.Devices) != 1 { // first call may block
		t.Fatalf("first call devices = %v", cfg.Devices)
	}
	for i := range 3 {
		cfg, d := get()
		if d > fast || len(cfg.Devices) != 1 {
			t.Errorf("cached call %d took %v with %d devices", i, d, len(cfg.Devices))
		}
	}
	if n := calls.Load(); n != 1 {
		t.Errorf("discovery ran %d times within the TTL, want 1", n)
	}

	time.Sleep(deviceCacheTTL + 50*time.Millisecond)
	if cfg, d := get(); d > fast || len(cfg.Devices) != 1 {
		t.Errorf("call after TTL took %v with %d devices, want stale list immediately", d, len(cfg.Devices))
	}
	deadline := time.Now().Add(2 * time.Second)
	for calls.Load() < 2 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if n := calls.Load(); n != 2 {
		t.Errorf("discovery ran %d times after the TTL, want a background refresh (2)", n)
	}
	// Let the refresh finish before cleanup restores the fakes.
	for time.Now().Before(deadline) {
		deviceCache.mu.Lock()
		done := !deviceCache.refreshing
		deviceCache.mu.Unlock()
		if done {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// fakeClusterRunner counts starts and stops instead of binding real ports.
func fakeClusterRunner() (r *clusterRunner, starts, stops *int) {
	starts, stops = new(int), new(int)
	r = newClusterRunner(context.Background(), &Scheduler{})
	r.start = func(context.Context, *Scheduler) func() {
		*starts++
		return func() { *stops++ }
	}
	return r, starts, stops
}

func TestClusterRunnerApply(t *testing.T) {
	clusterTestHome(t)
	r, starts, stops := fakeClusterRunner()
	set := func(kv map[string]any) {
		t.Helper()
		if err := envconfig.UpdateServerConfig(kv); err != nil {
			t.Fatal(err)
		}
		r.apply()
	}
	check := func(step string, wantStarts, wantStops int) {
		t.Helper()
		if *starts != wantStarts || *stops != wantStops {
			t.Errorf("%s: starts=%d stops=%d, want %d %d", step, *starts, *stops, wantStarts, wantStops)
		}
	}

	r.apply()
	check("off at startup", 0, 0)
	set(map[string]any{"cluster": true})
	check("turned on", 1, 0)
	set(map[string]any{"cluster_placement": "greedy"})
	check("placement is read per load, no restart", 1, 0)
	set(map[string]any{"cluster_share": false})
	check("share changed, restarted", 2, 1)
	set(map[string]any{"cluster": false})
	check("turned off", 2, 2)
}

func clusterConfigRequest(t *testing.T, s *Server, remoteAddr string, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/cluster/config", bytes.NewReader(b))
	c.Request.RemoteAddr = remoteAddr
	s.UpdateClusterConfigHandler(c)
	return w
}

func TestUpdateClusterConfigHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	clusterTestHome(t)
	r, starts, _ := fakeClusterRunner()
	s := &Server{cluster: r}
	on := true

	// X-Forwarded-For must not make a remote client look local.
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/cluster/config", bytes.NewReader([]byte(`{"enabled":true}`)))
	c.Request.RemoteAddr = "192.0.2.50:40000"
	c.Request.Header.Set("X-Forwarded-For", "127.0.0.1")
	s.UpdateClusterConfigHandler(c)
	if w.Code != http.StatusForbidden || envconfig.Cluster() {
		t.Fatalf("remote request: status %d, cluster=%v; want 403 and cluster still off", w.Code, envconfig.Cluster())
	}

	bad := "roundrobin"
	if w := clusterConfigRequest(t, s, "127.0.0.1:40000", api.ClusterConfigRequest{Placement: &bad}); w.Code != http.StatusBadRequest {
		t.Errorf("invalid placement: status %d, want 400", w.Code)
	}

	w = clusterConfigRequest(t, s, "[::1]:40000", api.ClusterConfigRequest{Enabled: &on})
	if w.Code != http.StatusOK {
		t.Fatalf("local request: status %d, body %s", w.Code, w.Body)
	}
	var got api.ClusterConfig
	if err := json.NewDecoder(w.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Enabled || got.Sources["enabled"] != "config" || *starts != 1 {
		t.Errorf("after on: %+v, starts=%d; want enabled from config, started once", got, *starts)
	}
	if len(got.Devices) != 1 || got.Devices[0].Name != "CUDA0" || got.Devices[0].Description != "Test GPU" || got.Devices[0].TotalMemory != 8<<30 {
		t.Errorf("devices = %+v, want the one local GPU", got.Devices)
	}
}

func TestFilterSharedDevices(t *testing.T) {
	devs := []ml.DeviceInfo{{Name: "CUDA0"}, {Name: "CUDA1"}}
	if got := filterSharedDevices(devs, nil); len(got) != 2 {
		t.Errorf("empty list should keep all, got %v", got)
	}
	if got := filterSharedDevices(devs, []string{"cuda1"}); len(got) != 1 || got[0].Name != "CUDA1" {
		t.Errorf("want only CUDA1, got %v", got)
	}
}

// cache_gb null in the config means no cap; cache_auto removes a saved one.
func TestUpdateClusterConfigCacheAuto(t *testing.T) {
	gin.SetMode(gin.TestMode)
	clusterTestHome(t)
	s := &Server{}
	post := func(body api.ClusterConfigRequest) api.ClusterConfig {
		t.Helper()
		w := clusterConfigRequest(t, s, "127.0.0.1:40000", body)
		var got api.ClusterConfig
		if w.Code != http.StatusOK || json.Unmarshal(w.Body.Bytes(), &got) != nil {
			t.Fatalf("status %d, body %s", w.Code, w.Body)
		}
		return got
	}
	gb, auto := uint(64), true

	if got := currentClusterConfig(context.Background()); got.CacheGB != nil {
		t.Errorf("default cache_gb = %d, want nil", *got.CacheGB)
	}
	if got := post(api.ClusterConfigRequest{CacheGB: &gb}); got.CacheGB == nil || *got.CacheGB != 64 {
		t.Errorf("after cache_gb=64: %v", got.CacheGB)
	}
	if w := clusterConfigRequest(t, s, "127.0.0.1:40000", api.ClusterConfigRequest{CacheGB: &gb, CacheAuto: &auto}); w.Code != http.StatusBadRequest {
		t.Errorf("cache_gb with cache_auto: status %d, want 400", w.Code)
	}
	if got := post(api.ClusterConfigRequest{CacheAuto: &auto}); got.CacheGB != nil || got.Sources["cache_gb"] != "default" {
		t.Errorf("after cache_auto: %v, source %q; want nil, default", got.CacheGB, got.Sources["cache_gb"])
	}
}
