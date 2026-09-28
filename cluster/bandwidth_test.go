package cluster

import (
	"context"
	"net"
	"testing"
)

func TestBandwidthRoundTrip(t *testing.T) {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go acceptBandwidthConns(ctx, ln)

	port := ln.Addr().(*net.TCPAddr).Port
	p := Peer{ID: "p1", Addr: "127.0.0.1"}
	tbl := &Table{peers: map[string]Peer{"p1": p}}

	bandwidthPeer(tbl, p, port)

	got := tbl.peers["p1"].BandwidthMbps
	if got <= 0 {
		t.Fatalf("BandwidthMbps = %v, want > 0", got)
	}
}

func TestBandwidthPeerUnreachableLeavesLastKnown(t *testing.T) {
	// Port 1 (TCPMUX) on loopback: nothing listens there, so this fails
	// fast with "connection refused" rather than waiting out
	// bandwidthTimeout, keeping the test quick.
	p := Peer{ID: "p1", Addr: "127.0.0.1", BandwidthMbps: 42}
	tbl := &Table{peers: map[string]Peer{"p1": p}}

	bandwidthPeer(tbl, p, 1)

	got := tbl.peers["p1"].BandwidthMbps
	if got != 42 {
		t.Fatalf("BandwidthMbps = %v, want unchanged 42", got)
	}
}
