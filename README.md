# LocalAI HyperQwen backend

A thin LocalAI gRPC adapter supervising a native HyperQwen HTTP server. HQ owns inference, launch profiles and startup tuning; the adapter handles request translation and process lifecycle. It does not modify LocalAI's stock backends or import native ML frameworks.

## Requirements

- Linux and Python 3.12.
- An existing HQ runtime matching `runtime.lock.json`, with its native dependencies and a supported NVIDIA GPU/driver for inference.
- An existing prepared checkpoint directory. No model downloads, conversion, quantization or model-repository script execution are performed by this adapter.

The metadata preflight accepts indexed compressed-tensors pack-quantized int4/int8 checkpoints declaring the supported Qwen3.5/3.8 architecture family, with tokenizer/template and selected speculation artifacts. It checks bounded metadata and file presence, not tensor contents; native HQ remains authoritative for actual model/hardware compatibility.

## Development

```sh
make setup
make test
```

Dependencies are hash-locked for Linux amd64 / CPython 3.12. Protocol bindings are generated from the pinned `protocol/backend.proto`, not committed. CPU tests need no Docker, GPU, controller executable or weights: a small synthetic HTTP fixture covers lifecycle and generation, and real subprocess tests cover parent crashes and TERM-resistant workers.

Workflow checks use `make lint-workflows` with actionlint 1.7.12 and ShellCheck on PATH. GitHub-hosted CPU CI fetches the checksum-pinned linter into temporary storage and runs the same tests; no image builds or model downloads occur in ordinary CI.

## Start the source adapter

With native HQ already installed:

```sh
HQ_ROOT=/absolute/path/to/hyperqwen \
  .venv/bin/python -m hyperqwen_backend.server --addr 127.0.0.1:50051
```

The gRPC endpoint is intended for a trusted LocalAI controller, not direct public exposure. Loading a model starts the selected native HQ launcher. Its HTTP service uses authenticated loopback readiness and a fixed internal served-model name. Controller-managed deployments should supply startup environment to the backend process; this source command starts a standalone development instance.

Use prepared model directories beneath LocalAI's configured models root when supplying relative model paths. Directory escapes are rejected. Keep weights read-only and provide writable HOME/cache/temp locations for native HQ/JIT. Stop the adapter gracefully to unload; its guardian also cleans up owned native workers if the adapter is killed, and controller-parent monitoring handles abrupt controller exit. Workers deliberately escaping the owned group or simultaneous guardian failure are not covered.

## Configuration

HQ startup settings pass through the backend process environment without a parallel configuration resolver:

- `HQ_ROOT`: trusted path to the installed HQ runtime (default `/app`).
- `HQ_PROFILE`: `single` (default) or `batch`.
- Native knobs such as SPEC, MAX_LEN, MAX_SEQS, GPU_UTIL and INT8_ACT remain HQ-owned. Explicit empty values are preserved; unset and empty are not interchangeable.
- `HQ_STARTUP_TIMEOUT` and `HQ_REQUEST_TIMEOUT`: finite positive seconds, default 600. HTTP request timeout is inactivity, not a total generation deadline.
- `HQ_GRPC_MAX_MESSAGE_MB`: integer 1–512, default 50.

The adapter reserves host, port, API authentication and internal model identity. EXTRA_ARGS cannot override those settings. Use HQ startup env rather than conflicting typed load settings or generic engine_args/options. Per-request sampling is translated separately.

For structured chat/tools, LocalAI model configuration should use `template.use_tokenizer_template: true` so structured messages reach the native tokenizer template rather than being flattened into a rendered prompt.

## Capabilities and limits

- Normal/streaming text and chat, reasoning, tools/continuation, sampling, usage and logprobs where supported by the protocol/frontend.
- Concurrent requests, upstream cancellation, load/unload/retry and explicit error handling.
- The pinned LocalAI legacy HTTP completion frontend drops usage/logprobs even when the adapter returns them. No controller workaround is included.
- No embeddings, multimodal or tokenization endpoints, automatic checkpoint preparation or blanket support for every HQ mode/hardware combination.

The protocol and tested controller pins are recorded separately in `runtime.lock.json`. CPU fixtures establish adapter behavior; they are not model quality/performance or GPU acceptance tests.
