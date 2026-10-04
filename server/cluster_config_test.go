package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

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
