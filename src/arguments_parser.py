import argparse
import os
from typing import List
import json
from pydantic import ValidationError
from .errors import (
    InputFileMissing,
    InputFileFormatError,
    ModelValidationError
)
from .models import Prompt, Function, Parameter, Type


class ArgumentsParser(argparse.ArgumentParser):  # Suppress argparse's own output by overriding error()
    args: argparse.Namespace

    def __init__(self) -> None:
        super().__init__()
        self.add_argument("--functions_definition", required=True)
        self.add_argument("--input", required=True)
        # self.add_argument("--output", required=True)
        self.args = self.parse_args()

    def get_functions(self) -> List[Function]:
        return ArgumentsParser.load_functions(
            ArgumentsParser.get_full_path(self.args.functions_definition)
        )

    def get_prompts(self) -> List[Prompt]:
        return ArgumentsParser.load_prompts(
            ArgumentsParser.get_full_path(self.args.input)
        )

    def error(self, message) -> None:
        raise SystemExit(message)

    @staticmethod
    def get_full_path(file_name: str) -> str:
        if os.path.dirname(file_name) == "":
            return "data/input/" + file_name
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
                            type=Type.model_validate(param_type)
                        )
                        for name, param_type in function["parameters"].items()
                    ],
                    returns=Type.model_validate(function["returns"])
                ))

            return functions
        except FileNotFoundError:
            raise InputFileMissing(file_name)
        except json.JSONDecodeError as e:
            raise InputFileFormatError(e.msg, e.lineno)
        except ValidationError as e:
            raise ModelValidationError(e.errors()[0]["input"])
