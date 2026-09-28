package server

import (
	"encoding/json"
	"net/http"
	"slices"
	"testing"
	"time"

	"github.com/borism/ollama-cluster/api"
	"github.com/borism/ollama-cluster/cluster"
	"github.com/borism/ollama-cluster/ml"
)

func TestClusterPeerToAPI(t *testing.T) {
	now := time.Now()
	got := clusterPeerToAPI(cluster.Peer{
		ID:      "abc123",
		Addr:    "192.0.2.1",
		Addrs:   []string{"192.0.2.1", "192.0.2.2"},
		RPCPort: 50052,
		Devices: []ml.DeviceInfo{
			{Name: "CUDA0", TotalMemory: 20 << 30, FreeMemory: 10 << 30},
		},
		Load:     0.5,
		Latency:  25 * time.Millisecond,
		LastSeen: now,
	})

	if got.ID != "abc123" || got.Addr != "192.0.2.1" {
		t.Errorf("id/addr not carried through: %+v", got)
	}
	if !slices.Equal(got.Addrs, []string{"192.0.2.1", "192.0.2.2"}) {
		t.Errorf("addrs not carried through: %+v", got.Addrs)
	}
	if !got.Sharing {
		t.Error("expected Sharing true when RPCPort is nonzero")
	}
	if got.LatencyMs != 25 {
		t.Errorf("expected LatencyMs 25, got %v", got.LatencyMs)
	}
	if !got.LastSeen.Equal(now) {
		t.Errorf("expected LastSeen %v, got %v", now, got.LastSeen)
	}
	if len(got.Devices) != 1 || got.Devices[0] != (api.ClusterDevice{Name: "CUDA0", TotalMemory: 20 << 30, FreeMemory: 10 << 30}) {
		t.Errorf("device not carried through correctly: %+v", got.Devices)
	}

	if notSharing := clusterPeerToAPI(cluster.Peer{RPCPort: 0}); notSharing.Sharing {
		t.Error("expected Sharing false when RPCPort is zero")
	}

	// A same-subnet peer's round trip is routinely sub-millisecond; that
	// must not truncate to the zero value omitempty then drops (see
	// api.ClusterPeer.LatencyMs).
	fast := clusterPeerToAPI(cluster.Peer{Latency: 600 * time.Microsecond})
	if fast.LatencyMs != 0.6 {
		t.Errorf("expected sub-ms LatencyMs 0.6, got %v", fast.LatencyMs)
	}
}

func TestClusterPeersHandlerDisabled(t *testing.T) {
	s := Server{sched: &Scheduler{}}

	w := createRequest(t, s.ClusterPeersHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	var resp api.ClusterListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	if resp.Enabled {
		t.Error("expected Enabled false when clusterTable is nil")
	}
	if len(resp.Peers) != 0 {
		t.Errorf("expected no peers, got %d", len(resp.Peers))
	}
}
