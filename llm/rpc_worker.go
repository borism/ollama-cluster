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
	exe  string
	args []string
	env  []string
	host string
	port int
	// stopCh is closed by Stop so a restart backoff sleep can end early.
	stopCh chan struct{}

	mu        sync.Mutex // guards the fields below
	cmd       *exec.Cmd
	done      chan struct{} // closed when cmd exits; replaced on restart
	status    *StatusWriter
	stopped   bool // set by Stop: an exit after this is deliberate
	startedAt time.Time
	backoff   time.Duration
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

// rpcDeviceArgs is the -d flag limiting ggml-rpc-server to devices, if any.
func rpcDeviceArgs(devices []string) []string {
	if len(devices) == 0 {
		return nil
	}
	return []string{"-d", strings.Join(devices, ",")}
}

// StartRPCWorker launches ggml-rpc-server so this instance can donate spare
// compute to another instance's model load over llama.cpp RPC. port=0 picks
// any free port -- call Port()/Addr() on the result to find out which one.
// cacheDir, when non-empty, turns on the worker's local tensor cache
// (-c/--cache) so a model already sent once over RPC isn't re-sent on every
// load. devices, when non-empty, limits the worker to those ggml devices
// (-d, e.g. "CUDA1"); otherwise it exposes every accelerator.
//
// Returns once the worker is actually accepting TCP connections, or once it
// exits or times out trying. ggml-rpc-server's only startup log line
// (ggml_backend_rpc_start_server, ggml/src/ggml-rpc/ggml-rpc.cpp) prints
// before the listening socket is bound, so it can't be used as a readiness
// signal -- a bounded connect-retry loop against the port is what's actually
// robust here.
func StartRPCWorker(port int, cacheDir string, devices []string) (*RPCWorker, error) {
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
	args = append(args, rpcDeviceArgs(devices)...)
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

	env := os.Environ()
	if cacheDir != "" {
		// rpc-server.cpp has no flag for the cache path itself -- it always
		// derives one from $LLAMA_CACHE (fs_get_cache_directory in
		// tools/rpc/rpc-server.cpp), appending "rpc/" to it.
		env = append(env, "LLAMA_CACHE="+cacheDir)
	}

	w := &RPCWorker{
		exe:    exe,
		args:   args,
		env:    env,
		host:   host,
		port:   port,
		stopCh: make(chan struct{}),
	}
	if err := w.spawn(); err != nil {
		return nil, err
	}

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

var errRPCWorkerStopped = errors.New("rpc worker stopped")

const (
	rpcRestartMinBackoff = time.Second
	rpcRestartMaxBackoff = time.Minute
	// A worker that stayed up this long before dying is treated as healthy
	// again: the next restart goes back to the minimum backoff.
	rpcRestartHealthyRun = time.Minute
)

// rpcRestartBackoff returns how long to wait before the next restart, given
// the previous wait (0 for none) and how long the worker ran before dying.
func rpcRestartBackoff(prev, ranFor time.Duration) time.Duration {
	if prev <= 0 || ranFor >= rpcRestartHealthyRun {
		return rpcRestartMinBackoff
	}
	return min(prev*2, rpcRestartMaxBackoff)
}

// spawn starts one ggml-rpc-server process with the worker's fixed args and
// env (so a restart keeps the same port) and watches it for exit.
func (w *RPCWorker) spawn() error {
	cmd := exec.Command(w.exe, w.args...)
	cmd.Env = w.env
	status := NewStatusWriter(nil)
	cmd.Stdout = status
	cmd.Stderr = status

	w.mu.Lock()
	defer w.mu.Unlock()
	if w.stopped {
		return errRPCWorkerStopped
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start ggml-rpc-server: %w", err)
	}
	done := make(chan struct{})
	w.cmd, w.done, w.status, w.startedAt = cmd, done, status, time.Now()

	go func(startedAt time.Time) {
		err := cmd.Wait()
		close(done)
		w.mu.Lock()
		stopped := w.stopped
		w.mu.Unlock()
		if stopped {
			slog.Info("cluster: rpc worker exited", "error", err, "last_output", status.LastError())
			return
		}
		// Port() (and so the beacon's Sharing flag) already reflects the
		// exit via w.running(), which is silent about *why* the worker went
		// away -- two real crashes this fleet hit left zero trace.
		//
		// TODO(upstream llama.cpp): both crashes this caught turned out to
		// be a clean exit (err == nil, no captured stderr) -- traced to
		// ggml_backend_rpc_start_server's serve loop
		// (ggml/src/ggml-rpc/ggml-rpc.cpp): a single failed accept() on the
		// listening socket logs "Failed to accept client connection" and
		// returns, ending the whole process, no retry. Restarting here
		// covers it; fixing the loop itself (loop past a transient
		// accept() failure instead of returning) is tracked upstream in
		// issue #25.
		slog.Warn("cluster: rpc worker exited unexpectedly, restarting", "error", err, "last_output", status.LastError())
		w.restart(time.Since(startedAt))
	}(w.startedAt)
	return nil
}

// restart waits out the backoff, then respawns the worker on the same port
// and waits for it to listen. If the new process dies or never listens, its
// own exit watcher calls restart again, so only a failed Start loops here.
func (w *RPCWorker) restart(ranFor time.Duration) {
	for {
		w.mu.Lock()
		w.backoff = rpcRestartBackoff(w.backoff, ranFor)
		backoff := w.backoff
		w.mu.Unlock()

		select {
		case <-w.stopCh:
			return
		case <-time.After(backoff):
		}
		err := w.spawn()
		if errors.Is(err, errRPCWorkerStopped) {
			return
		}
		if err != nil {
			slog.Warn("cluster: rpc worker restart failed", "error", err)
			ranFor = 0
			continue
		}
		// Same 2 min as the initial start (Metal compile on first run).
		if err := w.waitUntilListening(2 * time.Minute); err != nil {
			slog.Warn("cluster: restarted rpc worker is not listening", "error", err)
			w.mu.Lock()
			cmd := w.cmd
			w.mu.Unlock()
			// Its exit watcher takes it from here.
			_ = cmd.Process.Kill()
			return
		}
		slog.Info("cluster: rpc worker restarted", "addr", w.Addr())
		return
	}
}

// tidyCache runs tidyRPCCache every rpcCacheTidyInterval until Stop, across
// restarts. It can't turn a running worker's cache off, only free space for it.
func (w *RPCWorker) tidyCache(dir string, maxBytes uint64) {
	t := time.NewTicker(rpcCacheTidyInterval)
	defer t.Stop()
	for {
		select {
		case <-w.stopCh:
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
	w.mu.Lock()
	done, status := w.done, w.status
	w.mu.Unlock()
	deadline := time.Now().Add(timeout)
	addr := net.JoinHostPort("127.0.0.1", strconv.Itoa(w.port))

	for {
		select {
		case <-done:
			if msg := status.LastError(); msg != "" {
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
	if !w.stopped {
		w.stopped = true
		close(w.stopCh)
	}
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
