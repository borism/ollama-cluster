package cluster

import (
	"net"
	"strconv"
	"testing"
	"time"
)

// TestChooseAddrByLinkQuality: addresses with the same connect latency (a
// switch link and a direct cable) are told apart by bandwidth; a current
// address chosen that way (keep) survives a latency blip and a failed dial
// and is given up only for a much faster close rival; a slow-latency
// (Wi-Fi) address never wins on bandwidth.
func TestChooseAddrByLinkQuality(t *testing.T) {
	both := []probed{{"switch", 100 * time.Microsecond}, {"direct", 300 * time.Microsecond}}
	blip := []probed{{"switch", 100 * time.Microsecond}, {"direct", 5 * time.Millisecond}}
	onlySwitch := []probed{{"switch", 100 * time.Microsecond}}
	wifi := []probed{{"switch", 100 * time.Microsecond}, {"wifi", 5 * time.Millisecond}}
	good := map[string]float64{"switch": 860, "direct": 1850}
	cases := []struct {
		name    string
		got     []probed
		current string
		bw      map[string]float64
		keep    bool
		want    string
	}{
		{"faster link wins", both, "switch", good, false, "direct"},
		{"faster link wins, kept current much slower", both, "switch", good, true, "direct"},
		{"noise keeps current", both, "switch", map[string]float64{"switch": 940, "direct": 1000}, true, "switch"},
		{"noise keeps current (other way)", both, "direct", map[string]float64{"switch": 1000, "direct": 940}, true, "direct"},
		{"unmeasured keeps close current", both, "direct", nil, false, "direct"},
		{"unmeasured, current not close, latency", both, "other", nil, false, "switch"},
		{"far latency ignores bandwidth", blip, "", map[string]float64{"switch": 100, "direct": 2300}, false, "switch"},
		{"wifi never wins on bandwidth", wifi, "", map[string]float64{"switch": 100, "wifi": 5000}, false, "switch"},
		{"wifi current, not kept, moves to wired", wifi, "wifi", map[string]float64{"wifi": 5000}, false, "switch"},
		{"one latency blip keeps measured-best", blip, "direct", good, true, "direct"},
		{"one failed dial keeps measured-best", onlySwitch, "direct", good, true, "direct"},
		{"repeated failures give it up", onlySwitch, "direct", good, false, "switch"},
		{"much worse bandwidth gives it up", both, "direct", map[string]float64{"switch": 860, "direct": 300}, true, "switch"},
	}
	for _, c := range cases {
		if got := chooseAddr(c.got, c.current, c.bw, c.keep); got != c.want {
			t.Errorf("%s: chooseAddr = %q, want %q", c.name, got, c.want)
		}
	}
}

// loopbackPeerListener accepts and closes connections on every IPv4
// address, and returns its port. The test is skipped where 127.0.0.2 isn't
// a loopback address (macOS has no alias for it unless one is configured).
func loopbackPeerListener(t *testing.T) int {
	t.Helper()
	probe, err := net.Listen("tcp4", "127.0.0.2:0")
	if err != nil {
		t.Skipf("127.0.0.2 isn't a loopback address here: %v", err)
	}
	probe.Close()
	ln, err := net.Listen("tcp4", "0.0.0.0:0")
	if err != nil {
		t.Skipf("no loopback TCP available in this sandbox: %v", err)
	}
	t.Cleanup(func() { ln.Close() })
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
	c, err := net.DialTimeout("tcp4", net.JoinHostPort("127.0.0.2", strconv.Itoa(port)), time.Second)
	if err != nil {
		t.Skipf("127.0.0.2 isn't dialable here: %v", err)
	}
	c.Close()
	return port
}

// TestProbePeerPrefersFasterLink wires it through probePeer: two reachable
// candidates with near-identical latency, the other one measured faster,
// ends up as Addr with its bandwidth recorded.
func TestProbePeerPrefersFasterLink(t *testing.T) {
	port := loopbackPeerListener(t)

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

// TestProbePeerKeepsMeasuredBestThroughFailedDial: the current address won
// by measurement, then its dial fails (192.0.2.0/24 is non-routable). One
// failure leaves Addr alone; the linkFailLimit-th consecutive one moves it.
func TestProbePeerKeepsMeasuredBestThroughFailedDial(t *testing.T) {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
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

	for _, c := range []struct {
		priorFailures int
		want          string
	}{{0, "192.0.2.1"}, {linkFailLimit - 1, "127.0.0.1"}} {
		p := Peer{ID: "p1", Addr: "192.0.2.1", Addrs: []string{"192.0.2.1", "127.0.0.1"}, RPCPort: port, LastSeen: time.Now()}
		tbl := &Table{peers: map[string]Peer{"p1": p}}
		tbl.setLinkBandwidth("p1", "192.0.2.1", 1850)
		tbl.backoff = map[string]probeBackoff{"p1|192.0.2.1": {failures: c.priorFailures}}

		probePeer(tbl, p)

		if got := tbl.peers["p1"].Addr; got != c.want {
			t.Errorf("after %d prior failures: Addr = %q, want %q", c.priorFailures, got, c.want)
		}
	}
}
