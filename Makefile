ARGS ?=
INCLUDE_FILES := \
				src/__main__.py \
				src/errors.py \
				src/function_name_extractor.py \
				src/input_reader.py \
				src/json_extractors.py \
				src/llm.py \
				src/models.py \
				src/output_writer.py \
				src/parameters_extractor.py \
				src/prompt_pipeline.py

.PHONY: help install run lint clean debug

help:
	@echo "Commands:"
	@echo "make install			Installs development dependencies: mypy, flake8, numpy and pydantic"
	@echo "make run 			runs __main__.py"
	@echo "make debug			runs the main script in pdb"
	@echo "make lint			runs flake8 and mypy tests"
	@echo "make clean			cleans pycache mypy_cache"

install:
	@uv sync
	@echo "Dependencies installed"


run:
	@uv run python -m src $(ARGS)
# 	make run ARGS="--functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json"


debug:
	@uv run python -m pdb $(ARGS)


lint:
	@uv run flake8 $(INCLUDE_FILES)
	@uv run mypy \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--no-namespace-packages \
		--check-untyped-defs $(INCLUDE_FILES)


clean:
	@rm -rf __pycache__ .mypy_cache src/.mypy_cache src/__pycache__
	@echo "Cleaned build artifacts and cache files"
