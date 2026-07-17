ARGS ?=
INCLUDE_FILES := \
				input_parser.py \
				main.py

.PHONY: help install run lint lint-strict clean debug

help:
	@echo "Commands:"
	@echo "make install			Installs development dependencies: mypy, flake8, numpy and pydantic"
	@echo "make run 			runs main.py"
	@echo "make debug			runs the main script in pdb"
	@echo "make lint			runs flake8 and mypy tests"
	@echo "make lint-strict		runs flake8 and mypy --strict"
	@echo "make clean			cleans pycache mypy_cache"

install:
	@uv sync
	@uv pip install -e ./llm_sdk
	@echo "Dependencies installed"


run:
	@uv run python -m src $(ARGS)
# 	make run ARGS="--functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json"


debug:
	@uv run python -m pdb $(ARGS)


lint:
	@uv run flake8 $(INCLUDE_FILES)
	@uv run mypy \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs $(INCLUDE_FILES)


lint-strict:
	@uv run flake8 $(INCLUDE_FILES)
	@uv run mypy --strict $(INCLUDE_FILES)


clean:
	@rm -rf __pycache__ .mypy_cache ./src/__pycache__
	@echo "Cleaned build artifacts and cache files"
