PYTHON ?= python3
PIP := $(PYTHON) -m pip

ARGS ?= --help

.PHONY: install install-all test lint format run-web run-cli

install:
	$(PIP) install -e .[dev]

install-all:
	$(PIP) install -e .[dev,download,hmm,llm]

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

run-web:
	brainnet-web

run-cli:
	brainnet-cli $(ARGS)
