PYTHON ?= python3
PY := .venv/bin/python
ACTIONLINT ?= actionlint
IMAGE ?= localai-backend-hyperqwen:develop
PACKAGE ?= build/package
DOCKER_BUILD_FLAGS ?= --builder default
CONTROLLER ?= build/controller/local-ai-v4.12.1-linux-amd64

.PHONY: setup protocol test lint-workflows package extract smoke fetch-controller controller-test controller-package-test

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --require-hashes -r requirements.lock

protocol:
	$(PY) -m grpc_tools.protoc -I protocol --python_out=. --grpc_python_out=. protocol/backend.proto

test: protocol
	$(PY) -m unittest discover -s tests -v

lint-workflows:
	$(ACTIONLINT) .github/workflows/*.yml

package:
	docker build $(DOCKER_BUILD_FLAGS) --platform linux/amd64 --provenance=false --build-arg ADAPTER_REVISION=$$(git rev-parse HEAD)$$(git diff --quiet HEAD -- || printf '%s' -dirty) -t $(IMAGE) .

extract:
	sh packaging/extract.sh "$(IMAGE)" "$(PACKAGE)"

smoke:
	sh packaging/smoke.sh "$(PACKAGE)"

fetch-controller:
	$(PYTHON) -m tests.fetch_controller "$(CONTROLLER)"

controller-test: protocol
	$(PY) -m tests.controller_smoke "$(CONTROLLER)"

controller-package-test: protocol
	$(PY) -m tests.controller_smoke "$(CONTROLLER)" --package "$(PACKAGE)"
