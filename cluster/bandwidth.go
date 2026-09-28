package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"strconv"
	"sync"
	"time"
)

// bandwidthPayloadSize is how much data one bandwidth probe pushes through
// a peer's link. Big enough that TCP's initial slow-start window doesn't
// dominate the measurement on a fast link (Thunderbolt/10GbE), small
// enough that testing a peer costs at most a few hundred ms on any link
// worth using for RPC.
const bandwidthPayloadSize = 4 * 1024 * 1024 // 4 MiB

// bandwidthInterval is how often each peer's link is re-measured. Far
// longer than probeInterval (latency, a handshake-only dial) on purpose:
// this actually pushes bytes over the link, so running it that often would
// be exactly the "waste and delays" real workloads can't afford.
const bandwidthInterval = 5 * time.Minute

// bandwidthTimeout bounds one bandwidth probe end to end (dial, send,
// reply). 4 MiB at even a slow 10 Mbps link is ~3.2s; 10s leaves headroom
// without letting one struggling peer stall past its own probe (peers are
// measured concurrently, see bandwidthOnce, but each still needs its own
// ceiling).
const bandwidthTimeout = 10 * time.Second

// bandwidthResult is what the receiving end reports back over the same
// connection once it has read the whole payload: how many bytes arrived
// and how long that took, measured on the receiving side -- the side that
// actually experiences the receive window, and the direction weights flow
// in real RPC traffic (head to worker) -- not the sender's write-buffered
// view of its own throughput.
type bandwidthResult struct {
	Bytes int64 `json:"bytes"`
	Nanos int64 `json:"nanos"`
}

// bandwidthListen opens the bandwidth-probe listener. It shares
// Config.Port with the UDP discovery socket -- TCP and UDP are independent
// namespaces for the same port number, so this needs no port of its own to
// configure or document. The caller owns ln's lifecycle (see Start): it
// must be closed alongside the UDP socket, in the same place, so a
// restart on the same port can't race ahead of this one closing.
func bandwidthListen(port int) (net.Listener, error) {
	ln, err := net.Listen("tcp4", fmt.Sprintf(":%d", port))
	if err != nil {
		return nil, fmt.Errorf("cluster: listen tcp :%d: %w", port, err)
	}
	return ln, nil
}

func acceptBandwidthConns(ctx context.Context, ln net.Listener) {
	for {
		conn, err := ln.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
				slog.Debug("cluster: bandwidth accept", "error", err)
				return
			}
		}
		go serveBandwidthConn(conn)
	}
}

// serveBandwidthConn reads one probe's payload to EOF -- the client
// half-closes its write side once it has sent bandwidthPayloadSize bytes
// -- and reports back how long that took.
func serveBandwidthConn(conn net.Conn) {
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(bandwidthTimeout))
	start := time.Now()
	n, err := io.Copy(io.Discard, conn)
	if err != nil {
		slog.Debug("cluster: bandwidth serve", "error", err)
		return
	}
	result := bandwidthResult{Bytes: n, Nanos: time.Since(start).Nanoseconds()}
	json.NewEncoder(conn).Encode(result)
}

// bandwidthLoop periodically re-measures every peer's link throughput, on
// its own slower cadence (see bandwidthInterval).
func bandwidthLoop(ctx context.Context, t *Table, port int) {
	ticker := time.NewTicker(bandwidthInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			bandwidthOnce(t, port)
		}
	}
}

// bandwidthOnce measures every peer's currently-best address (see
// Peer.Addr) concurrently -- the one probeLoop already established has the
// lowest latency, not every candidate interface again, which would double
// the bytes pushed over the LAN for no placement-relevant benefit.
func bandwidthOnce(t *Table, port int) {
	var wg sync.WaitGroup
	for _, p := range t.Peers() {
		if p.Addr == "" {
			continue
		}
		wg.Add(1)
		go func(p Peer) {
			defer wg.Done()
			bandwidthPeer(t, p, port)
		}(p)
	}
	wg.Wait()
}

// bandwidthPeer pushes bandwidthPayloadSize bytes at p's address and
// records the throughput the far end reports back. A failure just leaves
// the peer's last-known BandwidthMbps in place (or zero, if never
// measured) and is retried next bandwidthInterval.
//
// ponytail: no backoff here, unlike probePeer's per-candidate one -- this
// already only runs once every bandwidthInterval per peer, so there's no
// hot loop to protect against. Add one if a chronically-unreachable peer's
// bandwidth port turns out to be a nuisance in practice.
func bandwidthPeer(t *Table, p Peer, port int) {
	addr := net.JoinHostPort(p.Addr, strconv.Itoa(port))
	conn, err := net.DialTimeout("tcp", addr, bandwidthTimeout)
	if err != nil {
		slog.Debug("cluster: bandwidth probe failed", "id", p.ID, "addr", addr, "error", err)
		return
	}
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(bandwidthTimeout))

	if _, err := io.CopyN(conn, zeroReader{}, bandwidthPayloadSize); err != nil {
		slog.Debug("cluster: bandwidth send", "id", p.ID, "addr", addr, "error", err)
		return
	}
	if tc, ok := conn.(*net.TCPConn); ok {
		tc.CloseWrite()
	}

	var result bandwidthResult
	if err := json.NewDecoder(conn).Decode(&result); err != nil || result.Nanos <= 0 {
		slog.Debug("cluster: bandwidth result", "id", p.ID, "addr", addr, "error", err)
		return
	}
	mbps := float64(result.Bytes*8) / (float64(result.Nanos) / 1e9) / 1e6
	slog.Debug("cluster: probed peer bandwidth", "id", p.ID, "addr", addr, "mbps", mbps)
	t.setBandwidth(p.ID, mbps)
}

// setBandwidth records a peer's last-measured link throughput. A no-op if
// the peer expired between listing and measuring, same as setBest.
func (t *Table) setBandwidth(id string, mbps float64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if p, ok := t.peers[id]; ok {
		p.BandwidthMbps = mbps
		t.peers[id] = p
	}
}

// zeroReader is an io.Reader that returns arbitrary (zero) bytes forever --
// content doesn't matter for a throughput test, only volume, so there's no
// need to allocate or reuse a real buffer for it.
type zeroReader struct{}

func (zeroReader) Read(b []byte) (int, error) {
	clear(b)
	return len(b), nil
}
