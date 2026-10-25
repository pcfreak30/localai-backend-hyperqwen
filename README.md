# LocalAI HyperQwen backend

Run HyperQwen models through LocalAI. A Python service accepts LocalAI's gRPC requests, starts HyperQwen and sends generation requests to its HTTP API. HyperQwen runs the model and controls its startup settings. LocalAI's stock backends are unchanged.

## Requirements

- Linux amd64. The package needs Ubuntu 24.04 or a host/container with compatible glibc 2.39, `/bin/sh`, bash and coreutils.
- Python 3.12 for source development, or Docker to build and extract the package. The package includes Python.
- A supported NVIDIA GPU and driver to run a model. Source development needs an installed HQ runtime matching `runtime.lock.json`; the package includes that runtime.
- An existing model directory prepared for HyperQwen. The adapter does not download, convert or quantize weights, or run scripts from model repositories.

Before starting HQ, the adapter checks the model configuration, tokenizer files and shard index. It accepts indexed compressed-tensors pack-quantized int4/int8 models in the Qwen3.5/3.8 family. It also checks the extra files used by multi-token prediction (MTP) when enabled. It does not read tensor contents; HQ loads the weights and reports model or hardware errors.

## Development

```sh
make setup
make test
```

Python dependencies have fixed versions and SHA256 hashes for Linux amd64 / CPython 3.12. `make test` generates protocol bindings from `protocol/backend.proto`; generated files are ignored by Git. Tests use a fake HTTP server and placeholder model metadata. They need no Docker, GPU, LocalAI executable or weights. Subprocess tests check worker cleanup after parent crashes, including workers that ignore TERM.

Run `make lint-workflows` with actionlint 1.7.12 and ShellCheck on PATH to check the GitHub workflows. CPU CI downloads the linter, verifies its SHA256 and runs the same tests. It does not build images or download models.

## Build the runtime package

The OCI image contains files for LocalAI to extract and execute. LocalAI runs its top-level `run.sh`. The image has no operating system and cannot run as a server container. It contains no weights or development test scripts.

```sh
make package                    # image: localai-backend-hyperqwen:develop
make extract                    # new directory: build/package
make smoke                      # check that extraction
```

The build uses the HQ image recorded in `runtime.lock.json` without reinstalling its ML dependencies. It copies Python, shared libraries, CUDA and the GCC/binutils headers and tools needed to compile kernels. Adapter dependencies stay separate from HQ's Python packages. Both lock files are included in the package.

Check free space in the workspace and Docker storage before building. Allow room for build copies as well as the image and extraction. The commands do not prune anything. Override `IMAGE` and `PACKAGE` to keep separate builds; extraction refuses an existing destination. If Docker bridge networking fails, use `DOCKER_BUILD_FLAGS='--builder default --network=host'`.

`make smoke` needs an existing `ubuntu:24.04` image. It runs with networking disabled, no GPU, a read-only root/package and writable temporary cache space. It checks imports, gRPC startup/shutdown, failed model loading, HQ's own dependency versions and launcher dry-runs. It also compiles C++, a Python extension and CUDA using the bundled compiler. These checks do not execute a GPU kernel or load real weights.

To download the LocalAI test executable, run `make fetch-controller`. It checks the SHA256 recorded in `runtime.lock.json` before installing it under `build/controller/`, or reuses a matching executable already there. It refuses to overwrite an existing file with the wrong hash.

You can also provide an existing executable:

```sh
make controller-test CONTROLLER=/absolute/path/to/local-ai
make controller-package-test CONTROLLER=/absolute/path/to/local-ai
```

These commands verify its checksum, then start a separate LocalAI instance on a loopback port with fake model files and a fake HQ server. Galleries and P2P are disabled. They do not download files, use a GPU or restart an existing service. Logs stay in ignored `build/`. The checks do not write Python bytecode into the package.

Register the extracted package with this LocalAI option:

```sh
--external-grpc-backends "hyperqwen:/absolute/path/to/package/run.sh"
```

LocalAI starts that executable and supplies the model YAML's environment settings. You can also place the entire package under the configured backends directory as `hyperqwen/run.sh` and its sibling directories. Copying or symlinking only the script will fail because it looks for dependencies beside it.

Select an idle GPU in the controller's environment. Keep the package and weights read-only where possible, with writable HOME/cache/temp directories. For tests, use separate workspace directories and an unused API port. Configure LocalAI authentication and TLS before exposing its API publicly. The Make commands do not publish images.

## Run from source

With HQ already installed:

```sh
HQ_ROOT=/absolute/path/to/hyperqwen \
  .venv/bin/python -m hyperqwen_backend.server --addr 127.0.0.1:50051
```

Keep this gRPC port private to LocalAI. A model load starts the selected HQ launcher. HQ's HTTP server listens on loopback with a generated API key; the adapter waits for the expected model before accepting generation requests. Set startup environment variables before starting this development command. A LocalAI-managed executable receives them from model YAML.

Relative model paths must stay beneath LocalAI's models root. Stop the adapter normally to unload the model. A separate guardian process stops its HQ workers if the adapter dies, and the adapter stops when its controller parent exits. This does not cover workers that leave the owned process group or a simultaneous guardian failure.

## Configuration

HQ reads startup settings from the backend process environment:

- `HQ_ROOT`: installed HQ directory, default `/app`.
- `HQ_PROFILE`: `single` (default) or `batch`.
- `SPEC`, `MAX_LEN`, `MAX_SEQS`, `GPU_UTIL`, `INT8_ACT` and other HQ settings pass through unchanged, except for the DFlash helper path resolution described below. Empty and unset values stay distinct.
- `HQ_STARTUP_TIMEOUT` and `HQ_REQUEST_TIMEOUT`: finite positive seconds, default 600. The HTTP timeout limits inactivity, not total generation time.
- `HQ_GRPC_MAX_MESSAGE_MB`: integer from 1 to 512, default 50.

The adapter sets the host, port, API key and internal model name. EXTRA_ARGS cannot override them. Configure startup through HQ environment variables rather than typed load settings or generic engine_args/options. Sampling settings come from each generation request.

Set `template.use_tokenizer_template: true` in LocalAI model YAML for chat and tools. This keeps messages structured so HQ's tokenizer template can handle them.

### DFlash helpers

With `HQ_PROFILE=single`, `SPEC=dflash2` and the default `DRAFT_METHOD=dflash`, an explicit `DRAFT` selects the additional helper model. Relative paths resolve beneath the target model directory, not the runtime directory. For example:

```yaml
env:
  HQ_PROFILE: "single"
  SPEC: "dflash2"
  DRAFT: "dflash"
  DFLASH_TOKENS: "7"
  LOOKUP: "1"
  ASYNC_SCHED: "0"
```

Place the prepared helper's `config.json` and single `model.safetensors` file in the target model's `dflash/` subdirectory. Absolute helper paths are also accepted. Relative paths, including directory symlinks, may not escape the target model directory; individual cache-file symlinks are supported.

The adapter checks the helper's DFlash2 architecture, metadata and nonempty payload file without opening tensors. Native HyperQwen still validates compatibility and loads the weights. There is no automatic download, conversion or execution of model-repository scripts. Batch mode does not use these helper settings. Leaving `DRAFT` unset preserves native helper discovery.

Model-relative `DRAFT` requires a runtime built with this feature. Older images, including the `b1c22fdd995f` image, do not resolve helper paths this way. Upgrade an existing backend installation before using a helper profile; installing a new model recipe is not a substitute for upgrading the runtime.

## Swift profiles and model gallery

Choose one of four uses, with or without an additional helper model. All eight profiles use the same prepared Swift model. Choose GPUs separately.

| Use | Without helper | With helper | Context limit | Conversations at once |
|---|---|---|---|---|
| Chat | `swift-chat` | `swift-chat-with-helper` | 32,768 tokens | 1 |
| Long Conversations | `swift-long-conversations` | `swift-long-conversations-with-helper` | 131,072 tokens | 1 |
| Multiple Conversations | `swift-multiple-conversations` | `swift-multiple-conversations-with-helper` | 32,768 tokens | Up to 4 |
| Multiple Long Conversations | `swift-multiple-long-conversations` | `swift-multiple-long-conversations-with-helper` | 131,072 tokens | Up to 4 |

The regular profiles use only Swift. The With Helper profiles add a separate 1.3 GB model to help generate replies. They need more GPU memory and are not always faster. In the configuration, these choices are `SPEC=mtp` and `SPEC=dflash2` respectively.

Both choices use `HQ_PROFILE=single`, which supports several conversations as well as one. The previous Multiple Conversations presets used the native batch launcher; it ignores `SPEC` and cannot offer this choice. The four-conversation limit remains, with `--enforce-eager` preserved for those profiles.

Limits include both input and replies. They are settings, not a promise that every GPU can hold that much. Conversations share available memory: four full-length inputs may not fit at once. Reduce `MAX_LEN` and `context_size` together, reduce `MAX_SEQS`, or share the model across more GPUs when memory is insufficient. These are experimental starting points, not quality or performance guarantees for every card.

Matching YAML files are under `examples/`. With existing prepared weights, copy one into LocalAI's models root and set `parameters.model` to the model directory's relative name. Give LocalAI writable cache/temp directories. Install or register the HyperQwen backend first.

The gallery uses one shared model directory under `hyperqwen/`. All recipes share the same 25 target files, including eight weight shards, pinned at upstream commit `2e6ad7769c0273e95813090acd147ca9a70c686f`. Helper recipes add two files from `syvai/Qwen3.8-27B-DFlash2-W4A16` at commit `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`. Every download has a SHA256 checksum. Listing recipes downloads no weights. A fresh install needs about 16.9 GB without the helper or 18.1 GB with it; files already present and matching their hashes can be reused.

The helper files are declared in the gallery's `files` list, so LocalAI downloads them during installation. Both gallery `files[].filename` and model-config `download_files[].filename` are relative to LocalAI's models root, not to `parameters.model`. The helper destinations therefore include the target directory:

```text
hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/dflash/config.json
hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/dflash/model.safetensors
```

The model YAML uses only `DRAFT: "dflash"`; the adapter resolves that beneath the target model. No fixed `/models` path is needed. We do not also add `download_files`, which would duplicate responsibility in model-config preloading. With existing weights and a manually copied example, place the two prepared helper files in the actual target model's `dflash/` subdirectory. Gallery installation only reuses weights already at its declared paths with matching hashes; it does not search other model folders.

Keep the full target-model path neutral: the pinned engine can override the regular mode when that path contains `dflash`, `dspark`, `eagle3` or `eagle-`. The gallery's Swift directory avoids those names. The helper itself belongs in its `dflash/` subdirectory.

Register the raw `gallery/models.yaml` URL at a chosen commit, keeping any other galleries configured. The new names replace `swift-hyperqwen-3090-graphs` and `swift-hyperqwen-3090-batch`. Existing installed YAMLs are not renamed automatically. Update client model names when switching, and stop the old profile before loading its replacement on the same GPUs.

### Choose GPUs for your deployment

Published profiles contain no GPU IDs, device count or machine-specific paths. They use one GPU unless you configure otherwise. Select idle GPUs explicitly before loading a model; the backend does not find spare GPUs or balance requests across model copies.

**One model across several GPUs:** expose the chosen cards to LocalAI, then add `CUDA_VISIBLE_DEVICES` and a matching `--tensor-parallel-size` to the model YAML's `env`. For example, if you deliberately selected two visible cards:

```yaml
env:
  CUDA_VISIBLE_DEVICES: "0,1"
  EXTRA_ARGS: "--tensor-parallel-size 2"
```

These are example device indices, not defaults to copy without checking your deployment. Change the selection and count together. For a Multiple Conversations profile, preserve its existing argument:

```yaml
  EXTRA_ARGS: "--enforce-eager --tensor-parallel-size 2"
```

Selected cards must support the model's split and be able to communicate inside the container. Provide sufficient container shared memory. More cards do not guarantee faster replies, and not every card count or combination is supported. The adapter forwards this setting to HyperQwen; it does not implement its own GPU distribution.

**Automatic sharing (requires a runtime containing this feature):** opt in with `HQ_GPU_SHARING: "auto"` in the model's `env`. The adapter counts CUDA devices visible to its native Python and adds the matching tensor-parallel setting. Container restrictions and `CUDA_VISIBLE_DEVICES` still apply. Explicit parallel settings in `EXTRA_ARGS` take precedence and skip detection. Detection failure or zero visible devices causes a configuration error, not a silent one-GPU fallback. The default, `HQ_GPU_SHARING: "off"`, leaves native behavior unchanged.

Visibility is not availability: automatic sharing uses every device exposed to the backend, including a busy one. Restrict visibility to devices allocated to this deployment first. It does not choose idle cards, route requests between model copies, guarantee a speedup or guarantee that arbitrary device counts and model dimensions are compatible. Native vLLM still validates the requested layout. Startup logs explain the selection; inspect vLLM's `tensor_parallel_size` and worker ranks to confirm it took effect.

**Cache budgeting:** the pinned DFlash launcher normally sets a fixed cache budget for one GPU. With tensor parallelism greater than one, that launcher already skips its fixed budget when `KV_MEM` is unset and uses `GPU_UTIL` instead. To request utilization-based sizing independently, set `HQ_CACHE_BUDGET: "auto"`; the adapter supplies an empty `KV_MEM` only if it was unset. An explicit `KV_MEM`, including an empty value, always wins. The default `HQ_CACHE_BUDGET: "native"` preserves native policy. An explicit native cache-size command-line argument remains authoritative. Utilization-based allocation is per worker, is not a combined pool of all GPU memory and can need extra headroom for compilation and transient allocations.

These opt-in policies require a new signed runtime; the previously published image does not implement them. They are not enabled in the gallery by this change, and automatic physical multi-GPU operation is not yet validated.

**Separate model copies:** copy a profile with a distinct `name` for each copy, select a different GPU through each copy's `CUDA_VISIBLE_DEVICES`, and leave out `--tensor-parallel-size`. Each copy needs memory for the whole model. Clients or an external router choose which model name receives each request; this backend does not automatically balance them.

Keep GPU selection in your deployment copies rather than the public gallery. HyperQwen chooses its own memory and attention defaults for the visible hardware. These profiles do not force an RTX-specific implementation or memory fraction. Override `GPU_UTIL` in your deployment if you need more memory headroom.

## Install the signed backend

Add this object to LocalAI's backend-gallery JSON list, keeping any galleries already configured:

```json
{
  "name": "hyperqwen",
  "url": "https://raw.githubusercontent.com/pcfreak30/localai-backend-hyperqwen/develop/gallery/backends.yaml",
  "verification": {
    "issuer": "https://token.actions.githubusercontent.com",
    "identity": "https://github.com/pcfreak30/localai-backend-hyperqwen/.github/workflows/package.yml@refs/heads/develop"
  }
}
```

LocalAI accepts this list through `LOCALAI_BACKEND_GALLERIES` or `--backend-galleries`. Replace `develop` in the catalog URL with a chosen commit SHA if you want the catalog to stay fixed. The catalog's image URL already uses a full SHA256 digest rather than a moving tag.

With that gallery configured:

```sh
local-ai backends install hyperqwen@hyperqwen --require-backend-integrity
```

If already installed, unload models using this backend before upgrading it, then reload them afterward:

```sh
local-ai backends upgrade hyperqwen --require-backend-integrity
```

Use the same configured backends path as your running instance. Installing a helper model recipe does not replace this upgrade step.

The verification policy is required for signed OCI installs. Keep strict integrity enabled; a policy with the wrong signing identity must fail. No GitHub login is needed to download the public image.

Set `TMPDIR` to an existing writable directory with room for the download and extraction. LocalAI stores compressed layers and a temporary tar there; a small `/tmp` can fill even when the backend directory has enough space. If GHCR returns an HTTP/2 `PROTOCOL_ERROR`, retry the install with `GODEBUG=http2client=0`. That changes the HTTP version without disabling signature checks.

This installs the runtime only. Select an example above to use existing prepared weights, or install a model recipe when you intend to download its weights. Choose a profile and helper option above, then select GPUs for your deployment.

Keep the previous signed image digest and gallery commit for rollback. If an update fails, use that catalog revision and stop the affected LocalAI instance before replacing its backend.

## Build and publish through GitHub

Large builds run only when requested. Pull requests continue to run CPU tests without building or publishing an image. No self-hosted runner is needed.

After the workflows are on `develop`, start a package build with `gh`:

```sh
gh workflow run package.yml --ref develop -f publish=false
gh run list --workflow package.yml
gh run watch RUN_ID
```

The job uses an Ubuntu 24.04 GitHub-hosted runner. It removes preinstalled Android, .NET, Haskell and CodeQL tools, then checks workspace and Docker storage. It requires 45 GiB free after that cleanup. The job stops if space is insufficient; it does not select a paid runner or register a self-hosted runner. Disk usage is printed after each large step. Its builder cache is removed before extraction to avoid keeping another runtime copy.

To push and sign an image after the package checks pass:

```sh
gh workflow run package.yml --ref develop -f publish=true
gh run watch RUN_ID
gh run download RUN_ID -n hyperqwen-candidate -D build/candidate
```

This publishes a test image in `ghcr.io/pcfreak30/localai-backend-hyperqwen`. Each run gets its own tag. The downloaded `candidate.json` records the image digest, source commit and signing identity. The source commit is stored in the image labels and covered by the signed image digest. Registry login uses GitHub's job token; signing uses GitHub OIDC and Cosign 2.6.5 with Sigstore bundles stored as OCI 1.1 referrers, as required by LocalAI 4.12.1. No personal token or signing key is required. New GHCR packages may need their visibility changed before anonymous users can install them.

Test that exact image digest with existing read-only weights on the intended GPU before approving it for the backend gallery. Check generation, streaming, tools, cancellation and worker cleanup, and check that GPU memory returns after shutdown. The build job's CPU checks do not perform those tests.

After reviewing the GPU results:

```sh
gh workflow run release.yml --ref develop \
  -f digest=sha256:IMAGE_DIGEST -f gpu_reviewed=true
gh run watch RELEASE_RUN_ID
gh run download RELEASE_RUN_ID -n hyperqwen-release -D build/release
```

Replace the run IDs and image digest with the actual values. The release job checks the build workflow's signature, reads the source commit from the signed image labels and creates backend-gallery files. It then moves `ghcr.io/pcfreak30/localai-backend-hyperqwen:latest` to that exact approved digest, checks that the digest is unchanged and verifies the build signature through the tag. It does not download image layers, rebuild, sign again, deploy or commit anything. The `gpu_reviewed` input is your acknowledgement, not an automated GPU check. Configure reviewers on GitHub's `release` environment if release approval should require another person.

Review the generated `backends.yaml` and `verification.json` before adding the backend entry and trusted signing policy to LocalAI. The trusted signing identity is `https://github.com/pcfreak30/localai-backend-hyperqwen/.github/workflows/package.yml@refs/heads/develop`, with issuer `https://token.actions.githubusercontent.com`. The gallery uses the build signature after GPU review. Approval does not add another signature because LocalAI 4.12.1 selects a bundle before checking the signing identity.

Test builds never move `latest`; only the reviewed release job does. Its artifacts include a digest-pinned `backends.yaml` for reproducibility and rollback, plus `backends-latest.yaml` for a moving update channel. The checked-in gallery remains digest-pinned until the first promotion has succeeded and the tag has passed LocalAI's strict verifier. LocalAI can detect digest changes for either form when checking for backend upgrades; a moving tag avoids needing a gallery commit for each approved runtime. Updates still require a backend upgrade, not just a model recipe refresh.

Neither workflow updates this repository's gallery or opens a PR. Check that intended users can pull and verify the signed image before adding it to the backend gallery. Keep the previous signed image digest and gallery commit for rollback; do not rebuild an old release to roll back.

## Supported requests

- Text and chat, including streaming, reasoning, tools and tool replies.
- Request sampling settings, token counts and token probabilities (`logprobs`) where LocalAI's API supports them.
- Concurrent requests, cancellation and model load/unload/retry.

The LocalAI version in `runtime.lock.json` drops usage/logprobs from its legacy HTTP completion responses even when the adapter returns them. The adapter does not change that frontend behavior.

Embeddings, multimodal and tokenization endpoints are not implemented. CPU tests check the adapter with fake responses; they do not measure GPU performance or answer quality.
