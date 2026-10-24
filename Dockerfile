# syntax=docker/dockerfile:1
ARG HQ_IMAGE=ghcr.io/syv-ai/hyperqwen@sha256:0acd4b9625473e51c9bab56406c30814bc0d0d7e0a38deb408c62664b5a77669
FROM ${HQ_IMAGE} AS builder
USER root
WORKDIR /adapter
# Keep adapter dependencies separate; never reinstall the HQ ML stack.
COPY requirements.lock .
RUN /app/venv/bin/python -m pip install --target=/adapter-deps --require-hashes -r requirements.lock
COPY protocol ./protocol
RUN PYTHONPATH=/adapter-deps /app/venv/bin/python -m grpc_tools.protoc -I protocol --python_out=. --grpc_python_out=. protocol/backend.proto
COPY packaging/package.py packaging/toolchain.py packaging/run.sh ./packaging/
COPY licenses ./licenses
COPY LICENSE THIRD_PARTY_NOTICES.txt runtime.lock.json ./
RUN /app/venv/bin/python packaging/package.py

FROM scratch AS backend
COPY --from=builder /package/ /
# Adapter edits do not invalidate the expensive runtime assembly.
COPY hyperqwen_backend /hyperqwen_backend/
ARG ADAPTER_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/pcfreak30/localai-backend-hyperqwen" \
      org.opencontainers.image.revision="${ADAPTER_REVISION}"
