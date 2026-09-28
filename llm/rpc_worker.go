package llm

import (
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/borism/ollama-cluster/envconfig"
)

// RPCWorker manages a running ggml-rpc-server subprocess: this Ollama
// instance donating spare local compute as a worker for another instance's
// llama.cpp RPC-backed inference (see tools/rpc/README.md /
// tools/rpc/rpc-server.cpp upstream).
type RPCWorker struct {
	mu     sync.Mutex
	cmd    *exec.Cmd
	host   string
	port   int
	done   chan struct{}
	status *StatusWriter
}

// FindRPCWorker locates the ggml-rpc-server binary in lib/ollama/, mirroring
// FindLlamaServer's search (llm/llama_binary.go).
func FindRPCWorker() (string, error) {
	path, candidates, err := findLlamaCppBinary("ggml-rpc-server", defaultLlamaCppBinarySearch())
	if err != nil {
		return "", fmt.Errorf("ggml-rpc-server binary not found (checked: %s)", strings.Join(candidates, ", "))
	}
	return path, nil
}

// StartRPCWorker launches ggml-rpc-server so this instance can donate spare
// compute to another instance's model load over llama.cpp RPC. port=0 picks
// any free port -- call Port()/Addr() on the result to find out which one.
// cacheDir, when non-empty, turns on the worker's local tensor cache
// (-c/--cache) so a model already sent once over RPC isn't re-sent on every
// load.
//
// Returns once the worker is actually accepting TCP connections, or once it
// exits or times out trying. ggml-rpc-server's only startup log line
// (ggml_backend_rpc_start_server, ggml/src/ggml-rpc/ggml-rpc.cpp) prints
// before the listening socket is bound, so it can't be used as a readiness
// signal -- a bounded connect-retry loop against the port is what's actually
// robust here.
func StartRPCWorker(port int, cacheDir string) (*RPCWorker, error) {
	exe, err := FindRPCWorker()
	if err != nil {
		return nil, err
	}

	if port == 0 {
		port, err = pickFreeTCPPort()
		if err != nil {
			return nil, fmt.Errorf("pick free port for ggml-rpc-server: %w", err)
		}
	}

	// ponytail: binds every interface (0.0.0.0) unconditionally so LAN peers
	// can actually reach it -- there's no caller yet that needs a narrower
	// bind. Add a host param if one shows up.
	host := "0.0.0.0"
	args := []string{"-H", host, "-p", strconv.Itoa(port)}
	// rpc-server.cpp puts its cache in $LLAMA_CACHE + "rpc/" (see below).
	rpcCache := filepath.Join(cacheDir, "rpc")
	maxCache := uint64(envconfig.ClusterCacheGB()) << 30
	if cacheDir != "" && maxCache > 0 {
		if err := os.MkdirAll(rpcCache, 0o755); err != nil {
			return nil, fmt.Errorf("create rpc cache dir: %w", err)
		}
		if tidyRPCCache(rpcCache, maxCache, time.Now(), diskFree) {
			args = append(args, "-c")
		} else {
			slog.Warn("cluster: under 10 GiB of disk free, sharing without the RPC tensor cache", "dir", rpcCache)
			cacheDir = ""
		}
	} else {
		cacheDir = ""
	}

	cmd := exec.Command(exe, args...)
	cmd.Env = os.Environ()
	if cacheDir != "" {
		// rpc-server.cpp has no flag for the cache path itself -- it always
		// derives one from $LLAMA_CACHE (fs_get_cache_directory in
		// tools/rpc/rpc-server.cpp), appending "rpc/" to it.
		cmd.Env = append(cmd.Env, "LLAMA_CACHE="+cacheDir)
	}

	status := NewStatusWriter(nil)
	cmd.Stdout = status
	cmd.Stderr = status

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start ggml-rpc-server: %w", err)
	}

	w := &RPCWorker{
		cmd:    cmd,
		host:   host,
		port:   port,
		done:   make(chan struct{}),
		status: status,
	}

	go func(cmd *exec.Cmd, done chan struct{}, status *StatusWriter) {
		err := cmd.Wait()
		close(done)
		// Port() (and so the beacon's Sharing flag) already reflects this
		// via w.running() -- but that's silent about *why* the worker went
		// away, which left two real crashes this fleet hit with zero trace
		// (see llm/rpc_worker_test.go). Stop() also lands here (it kills
		// the process, then waits on the same done channel this Wait()
		// closes), so a deliberate stop logs too -- an "exit status 1" from
		// a normal kill isn't worth telling apart from a real crash, at
		// slog.Info it's noise either way, not an alarm.
		slog.Info("cluster: rpc worker exited", "error", err, "last_output", status.LastError())
	}(cmd, w.done, status)

	// Generous on purpose: on Apple Silicon the first start after install
	// compiles the Metal kernel libraries before listening (~22s on an M2
	// Max; macOS caches them after). A dead process still fails fast.
	if err := w.waitUntilListening(2 * time.Minute); err != nil {
		w.Stop()
		return nil, err
	}

	if cacheDir != "" {
		go w.tidyCache(rpcCache, maxCache)
	}
	return w, nil
}

// tidyCache runs tidyRPCCache every rpcCacheTidyInterval until the worker
// exits. It can't turn a running worker's cache off, only free space for it.
func (w *RPCWorker) tidyCache(dir string, maxBytes uint64) {
	t := time.NewTicker(rpcCacheTidyInterval)
	defer t.Stop()
	for {
		select {
		case <-w.done:
			return
		case <-t.C:
			if !tidyRPCCache(dir, maxBytes, time.Now(), diskFree) {
				slog.Warn("cluster: under 10 GiB of disk free even with the RPC tensor cache emptied", "dir", dir)
			}
		}
	}
}

// waitUntilListening polls the worker's port until a TCP connection
// succeeds, the process exits, or timeout elapses.
func (w *RPCWorker) waitUntilListening(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	addr := net.JoinHostPort("127.0.0.1", strconv.Itoa(w.port))

	for {
		select {
		case <-w.done:
			if msg := w.status.LastError(); msg != "" {
				return fmt.Errorf("ggml-rpc-server exited before it started listening: %s", msg)
			}
			return errors.New("ggml-rpc-server exited before it started listening")
		default:
		}

		if conn, err := net.DialTimeout("tcp", addr, 200*time.Millisecond); err == nil {
			conn.Close()
			return nil
		}

		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for ggml-rpc-server to listen on %s", addr)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// Addr returns "host:port" the worker is actually listening on, or "" if
// it's not running.
func (w *RPCWorker) Addr() string {
	if w == nil || !w.running() {
		return ""
	}
	return net.JoinHostPort(w.host, strconv.Itoa(w.port))
}

// Port returns the port the worker is actually listening on, or 0 if it's
// not running.
func (w *RPCWorker) Port() int {
	if w == nil || !w.running() {
		return 0
	}
	return w.port
}

func (w *RPCWorker) running() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.cmd == nil {
		return false
	}
	select {
	case <-w.done:
		return false
	default:
		return true
	}
}

// Stop kills the worker process and waits for it to exit.
func (w *RPCWorker) Stop() error {
	w.mu.Lock()
	cmd := w.cmd
	done := w.done
	w.mu.Unlock()

	if cmd == nil || cmd.Process == nil {
		return nil
	}
	if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		return err
	}
	if done != nil {
		<-done
	}
	return nil
}

// pickFreeTCPPort asks the OS for an unused loopback port. Same
// listen-then-close approach startLlamaServer uses (llm/llama_server.go) --
// there's an inherent race between closing the probe listener and the real
// process binding the port, but it's the same tradeoff already accepted
// there, and simpler than plumbing an already-open listener fd into a
// subprocess just to avoid it.
func pickFreeTCPPort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}
