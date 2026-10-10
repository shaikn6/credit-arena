PYTHON ?= python

.PHONY: data run study test

data:  ## download both datasets and verify their checksums
	$(PYTHON) download_data.py

run:  ## main comparison: writes results.json, test_probs.npz and timings.json
	$(PYTHON) run.py

study:  ## repeated-split study on both datasets (about 7 minutes): writes study.json
	$(PYTHON) study.py

test:
	$(PYTHON) -m ruff check --select E9,F .
	$(PYTHON) -m pytest -q
