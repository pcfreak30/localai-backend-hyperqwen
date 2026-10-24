# LocalAI HyperQwen backend

A thin LocalAI gRPC adapter supervising a native HyperQwen HTTP server. HQ owns inference, launch profiles and startup tuning; the adapter handles request translation and process lifecycle. It does not modify LocalAI's stock backends or import native ML frameworks.

## Requirements

- Linux amd64. The extracted package targets Ubuntu 24.04/glibc 2.39-compatible hosts/containers with `/bin/sh`, bash and coreutils; it bundles Python and its runtime/compiler dependencies.
- Python 3.12 for source development, or Docker to build/extract the package.
- A supported NVIDIA GPU/driver for inference. Source development needs an existing HQ runtime matching `runtime.lock.json`; the standalone package bundles that pinned runtime.
- An existing prepared checkpoint directory. No model downloads, conversion, quantization or model-repository script execution are performed by this adapter.

The metadata preflight accepts indexed compressed-tensors pack-quantized int4/int8 checkpoints declaring the supported Qwen3.5/3.8 architecture family, with tokenizer/template and selected speculation artifacts. It checks bounded metadata and file presence, not tensor contents; native HQ remains authoritative for actual model/hardware compatibility.

## Development

```sh
make setup
make test
```

Dependencies are hash-locked for Linux amd64 / CPython 3.12. Protocol bindings are generated from the pinned `protocol/backend.proto`, not committed. CPU tests need no Docker, GPU, controller executable or weights: a small synthetic HTTP fixture covers lifecycle and generation, and real subprocess tests cover parent crashes and TERM-resistant workers.

Workflow checks use `make lint-workflows` with actionlint 1.7.12 and ShellCheck on PATH. GitHub-hosted CPU CI fetches the checksum-pinned linter into temporary storage and runs the same tests; no image builds or model downloads occur in ordinary CI.

## Standalone runtime package

The OCI image is an **extractable executable package**, not a runnable server container or full root filesystem. LocalAI executes its top-level `run.sh`. No nested Docker, model weights or development probes are shipped.

```sh
make package                    # local image: localai-backend-hyperqwen:develop
make extract                    # fresh directory: build/package
make smoke                      # checks the existing extraction; never re-extracts
```

The build reuses the digest-pinned HQ stack without reinstalling its ML dependencies. It relocates Python/native libraries, CUDA, GCC/binutils/sysroot and Python headers needed for native JIT. Adapter dependencies stay separate from HQ's Python environment. `runtime.lock.json` and `requirements.lock` are retained in the package.

Check workspace and Docker storage before building: the runtime is roughly 10 GB, and build layers plus extraction require substantially more. There is no automatic pruning. Use `IMAGE`/`PACKAGE` to retain multiple candidates; extraction refuses any existing destination. `DOCKER_BUILD_FLAGS='--builder default --network=host'` is available if Docker bridge networking fails.

`make smoke` requires a pre-existing `ubuntu:24.04` image. It uses no network or GPU, mounts the package/root read-only and supplies writable temporary cache space. It verifies imports, gRPC health/load rejection/free/shutdown, native dependency isolation, both native launcher dry-runs and C++/Python-extension/CUDA compilation without a host compiler. Compiler success is not GPU kernel execution or model acceptance.

For controller integration, provide the executable pinned in `runtime.lock.json`:

```sh
make controller-test CONTROLLER=/absolute/path/to/local-ai
make controller-package-test CONTROLLER=/absolute/path/to/local-ai
```

These checks verify the controller checksum before starting an isolated loopback instance with synthetic checkpoint/HTTP fixtures and galleries/P2P disabled. They do not download the controller or weights, allocate a GPU, or touch an existing LocalAI service. Logs stay under ignored `build/`; package checks do not write interpreter caches into the extraction.

Register the **entire extracted package** as a trusted LocalAI-managed executable, using this LocalAI option:

```sh
--external-grpc-backends "hyperqwen:/absolute/path/to/package/run.sh"
```

This is a local executable path, not a remote gRPC endpoint. LocalAI starts it and applies model YAML env. Alternatively install the entire package at the configured backends path as `hyperqwen/run.sh` and its sibling runtime directories. Do not copy or symlink only the script: dependencies are relative to the package root.

Select an approved idle GPU explicitly in the controller environment; do not default to an occupied GPU. Keep package/weights read-only where practical and provide writable HOME/cache/temp paths. Use an isolated workspace/unused API port for tests; configure LocalAI authentication/TLS separately before public exposure. The Make targets do not publish images or modify stock LocalAI backends.

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
