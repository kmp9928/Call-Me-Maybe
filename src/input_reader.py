import argparse
import json
import os
from typing import List
from pydantic import ValidationError
from .errors import (
    InputFileMissing,
    InputFileFormatError,
    ModelValidationError
)
from .models import Prompt, Function, Parameter


class InputReader(argparse.ArgumentParser):  # Suppress argparse's own output by overriding error()
    args: argparse.Namespace

    def __init__(self) -> None:
        super().__init__()
        self.add_argument("--functions_definition", required=True)
        self.add_argument("--input", required=True)
        self.add_argument("--output", required=True)
        self.args = self.parse_args()

    def get_functions(self) -> List[Function]:
        return InputReader.load_functions(
            InputReader.get_full_path(self.args.functions_definition, "input")
        )

    def get_prompts(self) -> List[Prompt]:
        return InputReader.load_prompts(
            InputReader.get_full_path(self.args.input, "input")
        )

    def get_output_file(self) -> str:
        return InputReader.get_full_path(self.args.output, "output")

    def error(self, message) -> None:
        raise SystemExit(message)

    @staticmethod
    def get_full_path(file_name: str, directory: str) -> str:
        if os.path.dirname(file_name) == "":
            return f"data/{directory}/" + file_name
        else:
            return file_name

    @staticmethod
    def load_prompts(file_name: str) -> List[Prompt]:
        prompts: List[Prompt] = []

        try:
            with open(file_name, "r") as file:
                data = json.load(file)

            for prompt in data:
                prompts.append(Prompt.model_validate(prompt))

            return prompts
        except FileNotFoundError:
            raise InputFileMissing(file_name)
        except json.JSONDecodeError as e:
            raise InputFileFormatError(e.msg, e.lineno)
        except ValidationError as e:
            raise ModelValidationError(e.errors()[0]["input"])

    @staticmethod
    def load_functions(file_name: str) -> List[Function]:
        functions: List[Function] = []

        try:
            with open(file_name, "r") as file:
                data = json.load(file)

            for function in data:
                functions.append(Function(
                    name=function["name"],
                    description=function["description"],
                    parameters=[
                        Parameter(
                            name=name,
                            type=param_type["type"]
                        )
                        for name, param_type in function["parameters"].items()
                    ],
                    returns=function["returns"]
                ))

            return functions
        except FileNotFoundError:
            raise InputFileMissing(file_name)
        except json.JSONDecodeError as e:
            raise InputFileFormatError(e.msg, e.lineno)
        except ValidationError as e:
            raise ModelValidationError(e.errors()[0]["input"])
