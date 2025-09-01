# Convenience Makefile for neuromorphic cortical column project

VENV=./venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
PYTEST=$(PY) -m pytest

.PHONY: help venv install upgrade test demo sim simple clean exp-step exp-freq exp-noise exp-stability exp-l23 exp-pwm exp-l4map

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
	@echo "  exp-stability - long-run stability experiment"
	@echo "  exp-l23  - L2/3 spectral radius over learning"
	@echo "  exp-pwm  - PWM duty monotonicity sweep"
	@echo "  exp-l4map- L4 selectivity mapping"

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

exp-stability:
	$(PY) -m experiments.runner stability --steps 10000 --size 64

exp-l23:
	$(PY) -m experiments.runner l23_spectral --steps 5000 --size 32

exp-pwm:
	$(PY) -m experiments.runner pwm --blocks 11 --size 32

exp-l4map:
	$(PY) -m experiments.runner l4map --steps 1500 --size 32
