# Convenience Makefile for neuromorphic cortical column project

VENV=./venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
PYTEST=$(PY) -m pytest

.PHONY: help venv install upgrade test demo sim simple clean

help:
	@echo "Targets:"
	@echo "  venv     - create virtualenv"
	@echo "  install  - install requirements into venv"
	@echo "  upgrade  - upgrade pip and reinstall requirements"
	@echo "  test     - run pytest"
	@echo "  demo     - run demo_cortical_column.py"
	@echo "  sim      - run cortical_column.py"
	@echo "  simple   - run simple_example.py"
	@echo "  clean    - remove venv and __pycache__"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PY) -m pip install -U pip
	$(PIP) install -r requirements.txt

upgrade: venv
	$(PY) -m pip install -U pip
	$(PIP) install -U -r requirements.txt

test:
	$(PYTEST) -q

demo:
	$(PY) demo_cortical_column.py

sim:
	$(PY) cortical_column.py

simple:
	$(PY) simple_example.py

clean:
	rm -rf $(VENV)
	rm -rf __pycache__ */__pycache__
