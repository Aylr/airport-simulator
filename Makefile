.PHONY: clean clean-test clean-pyc clean-build docs help installl github test release notebooks
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys


HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDCOLOR = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

print(OKBLUE + "You can run these make commands:\n" + ENDCOLOR)

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print(OKGREEN + "%-30s " % target + ENDCOLOR + "%s" % (help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help: ## Get help messages
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

github: ## Open the GitHub repo for this project
	open https://github.com/Aylr/airport-simulator

install: ## update your python environment
	pip install -r requirements.txt || true
	pip install -r dev-requirements.txt || true
	pip install -e ./ || true
	pip list

test: ## Run tests
	pytest -s tests

lint: ## Lint the entire project using black
	black . --exclude venv

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## Remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

dist: clean ## Builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

venv: ## Create a new virtualenv in the repo root
	rm -rf venv || true
	virtualenv venv
.PHONY: venv

notebooks: ## Run juypter lab for your notebooks
	jupyter lab

coverage: ## Check code coverage
	coverage run --source simulator -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html
