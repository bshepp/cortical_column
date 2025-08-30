# Convenience Makefile for neuromorphic cortical column project

VENV=./venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
PYTEST=$(PY) -m pytest

.PHONY: help venv install upgrade test demo sim simple clean exp-step exp-freq exp-noise

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
	@echo "  exp-step - run step response experiment"
	@echo "  exp-freq - run frequency sweep experiment"
	@echo "  exp-noise- run noise sweep experiment"

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

exp-step:
	$(PY) -m experiments.runner step --steps 2000 --size 64

exp-freq:
	$(PY) -m experiments.runner freq --steps 2000 --size 64

exp-noise:
	$(PY) -m experiments.runner noise --steps 500 --size 64
