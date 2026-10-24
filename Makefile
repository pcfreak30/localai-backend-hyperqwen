PYTHON ?= python3
PY := .venv/bin/python
ACTIONLINT ?= actionlint

.PHONY: setup protocol test lint-workflows

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --require-hashes -r requirements.lock

protocol:
	$(PY) -m grpc_tools.protoc -I protocol --python_out=. --grpc_python_out=. protocol/backend.proto

test: protocol
	$(PY) -m unittest discover -s tests -v

lint-workflows:
	$(ACTIONLINT) .github/workflows/*.yml
