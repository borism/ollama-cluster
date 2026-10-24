package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/borism/ollama-cluster/api"
)

func TestPrintIncompatiblePeers(t *testing.T) {
	var buf bytes.Buffer
	printIncompatiblePeers(&buf, []api.ClusterPeer{{ID: "ok", Compatible: true}})
	if buf.Len() != 0 {
		t.Fatalf("compatible peers should print nothing, got %q", buf.String())
	}

	printIncompatiblePeers(&buf, []api.ClusterPeer{
		{ID: "node-a", Addr: "10.0.0.2", Reason: "speaks RPC protocol 7.0, this node needs 8.0: upgrade the peer to match"},
		{ID: "node-b", Addr: "10.0.0.3", Version: "0.40.8-cluster.1", Reason: "speaks RPC protocol 9.0, this node needs 8.0: upgrade this node to match"},
	})
	out := buf.String()
	for _, want := range []string{"node-a", "unknown", "upgrade the peer to match", "node-b", "0.40.8-cluster.1", "upgrade this node to match"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
}
