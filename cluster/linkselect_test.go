package cluster

import (
	"net"
	"testing"
	"time"
)

// TestChooseAddrByLinkQuality: two addresses with the same connect latency
// (a switch link and a direct cable) are told apart by bandwidth, and the
// current one is kept through small differences (no flipping on noise).
func TestChooseAddrByLinkQuality(t *testing.T) {
	both := []probed{{"switch", 100 * time.Microsecond}, {"direct", 300 * time.Microsecond}}
	slow := []probed{{"switch", 100 * time.Microsecond}, {"direct", 5 * time.Millisecond}}
	cases := []struct {
		name    string
		got     []probed
		current string
		bw      map[string]float64
		want    string
	}{
		{"faster link wins", both, "switch", map[string]float64{"switch": 940, "direct": 2300}, "direct"},
		{"faster link wins from unset", both, "", map[string]float64{"switch": 940, "direct": 2300}, "direct"},
		{"noise keeps current", both, "switch", map[string]float64{"switch": 940, "direct": 1000}, "switch"},
		{"noise keeps current (other way)", both, "direct", map[string]float64{"switch": 1000, "direct": 940}, "direct"},
		{"unmeasured keeps current", both, "direct", nil, "direct"},
		{"unmeasured, current not close, latency", both, "other", nil, "switch"},
		{"far latency ignores bandwidth", slow, "direct", map[string]float64{"switch": 100, "direct": 2300}, "switch"},
	}
	for _, c := range cases {
		if got := chooseAddr(c.got, c.current, c.bw); got != c.want {
			t.Errorf("%s: chooseAddr = %q, want %q", c.name, got, c.want)
		}
	}
}

// TestProbePeerPrefersFasterLink wires it through probePeer: two reachable
// candidates with near-identical latency, the other one measured faster,
// ends up as Addr with its bandwidth recorded.
func TestProbePeerPrefersFasterLink(t *testing.T) {
	ln, err := net.Listen("tcp4", "0.0.0.0:0")
	if err != nil {
		t.Skipf("no loopback TCP available in this sandbox: %v", err)
	}
	defer ln.Close()
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}()
	port := ln.Addr().(*net.TCPAddr).Port

	p := Peer{ID: "p1", Addr: "127.0.0.1", Addrs: []string{"127.0.0.1", "127.0.0.2"}, RPCPort: port, LastSeen: time.Now()}
	tbl := &Table{peers: map[string]Peer{"p1": p}, bwPort: 1}
	tbl.setLinkBandwidth("p1", "127.0.0.1", 940)
	tbl.setLinkBandwidth("p1", "127.0.0.2", 2300)

	probePeer(tbl, p)

	got := tbl.peers["p1"]
	if got.Addr != "127.0.0.2" || got.BandwidthMbps != 2300 {
		t.Fatalf("Addr = %q, BandwidthMbps = %v, want the faster link 127.0.0.2 at 2300", got.Addr, got.BandwidthMbps)
	}
}
