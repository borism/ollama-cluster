# llama.cpp RPC fixes

This directory holds fixes to llama.cpp's RPC backend and `ggml-rpc-server`
that this fork carries until upstream merges them (#51). Each fix is its own
patch, so any one can be dropped when `LLAMA_CPP_VERSION` reaches a release
that has it.

## How they are applied

`rpc-patches.cmake` defines `OLLAMA_LLAMA_CPP_RPC_PATCH_COMMAND`. It runs
right after the `llama/compat/` patch command, everywhere that one runs:

- `llama/server/CMakeLists.txt`: the FetchContent `PATCH_COMMAND`, and the
  `FETCHCONTENT_SOURCE_DIR_LLAMA_CPP` source override.
- `cmake/local.cmake`: the superbuild's `ollama-llama-cpp-source` step.

Like the compat patch, it is skipped when `OLLAMA_LLAMA_CPP_SOURCE` points at
a developer's own tree. Both use `cmake/apply-git-patches.cmake`, which
applies every `*.patch` here in numeric filename order and skips a patch that
is already applied (`git apply --reverse --check`).

| Patch | Fix | Upstream | Tracking |
|---|---|---|---|
| `001-rpc-cache-no-truncated-files.patch` | A failed or interrupted cache write left a short or empty file that was then served as a cache hit. | [ggml-org/llama.cpp#30648](https://github.com/ggml-org/llama.cpp/pull/30648) (fixes #30633) | borism/llama.cpp#2 |
| `002-rpc-accept-retry-transient-errors.patch` | One failed `accept()` (a client that reset first, `EINTR`, `EMFILE`) ended the whole rpc-server process. | not submitted yet | borism/llama.cpp#3, #82 |

## Adding a patch

- Write one fix per file, named `NNN-rpc-<what>.patch` with the next unused
  number. Do not renumber the others when one is dropped.
- Make it with `git diff` in a clean checkout of the tag in
  `LLAMA_CPP_VERSION`. Above the diff, write a short header: what it fixes,
  an `Upstream:` line (the PR link, or "not submitted yet") and a
  `Tracking:` line. `git apply` ignores text before the first `diff --git`.
- Each patch must apply on its own to the clean tag, not only on top of the
  ones before it. That is what lets any one be dropped, and CI checks it.
- Add a row to the table above and a box to #51.

## Dropping a patch

Drop a patch when the pinned llama.cpp has the fix, through the upstream PR
or an equivalent change. Delete the file and its table row, and tick its box
in #51 with the upstream link. You will see one of these:

- The `patches` job in `.github/workflows/test.yaml` fails with
  "does not apply to <tag>".
- The configure prints `llama/rpc-patches: NNN-... already applied, skipping`.
- `git apply --reverse --check` of the patch succeeds in a clean checkout.

If a patch stops applying because the code around it moved but upstream does
not have the fix, regenerate it against the new tag instead.

## Checking a llama.cpp bump

The `patches` CI job configures `llama/server`, which applies every patch,
then checks each RPC patch against the clean tag. To check a tag by hand
before bumping:

```sh
git clone --depth 1 --branch <tag> https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
cd /tmp/llama.cpp
for p in /path/to/ollama/llama/rpc-patches/*.patch; do git apply --check "$p" || echo "FAIL $p"; done
git apply --check /path/to/ollama/llama/compat/*.patch /path/to/ollama/llama/rpc-patches/*.patch
```

The last line checks the whole series in the order the build applies it.
