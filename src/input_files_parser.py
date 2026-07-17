from typing import List
import json
from pydantic import ValidationError
from .errors import (
    InputFileMissing,
    InputFileFormatError,
    ModelValidationError
)
from .models import Prompt, Function, Parameter, Type


class InputParser:
    @staticmethod
    def load(file_name: str) -> List[Prompt]:
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


class FunctionDefinitionParser:
    @staticmethod
    def load(file_name) -> List[Function]:
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
