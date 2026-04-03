PYTHON ?= python3

.PHONY: setup download preprocess evaluate simulate tune demo-data notebook

setup:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) -m pokecoach.cli download

preprocess:
	$(PYTHON) -m pokecoach.cli preprocess

evaluate:
	$(PYTHON) -m pokecoach.cli evaluate

simulate:
	$(PYTHON) -m pokecoach.cli simulate --tier gen9vgc2024regg --mode smoke

tune:
	$(PYTHON) -m pokecoach.cli tune

demo-data:
	$(PYTHON) -m pokecoach.cli demo-data

notebook:
	$(PYTHON) -m pokecoach.cli make-notebook
