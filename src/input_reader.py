import argparse
import json
import os
from typing import List, Never
from pydantic import ValidationError
from .errors import (
    CommandLineArgumentsError,
    InputFileMissing,
    InputFileFormatError,
    ModelValidationError
)
from .models import Prompt, Function, Parameter


class InputReader(argparse.ArgumentParser):
    """Command-line argument parser and input file loader.

    Extends `argparse.ArgumentParser` to automatically parse required
    command-line arguments and load function definitions, prompts and output
    paths from disk.

    Attributes:
        args (argparse.Namespace): Parsed command-line arguments containing
            `functions_definition`, `input`, and `output`.
    """

    args: argparse.Namespace

    def __init__(self) -> None:
        """Initializes the InputReader and parses required arguments."""
        super().__init__()
        self.add_argument("--functions_definition", required=True)
        self.add_argument("--input", required=True)
        self.add_argument("--output", required=True)
        self.args = self.parse_args()

    def get_functions(self) -> List[Function]:
        """Loads and returns the list of functions from the specified path.

        Returns:
            List[Function]: A list of instantiated `Function` objects.

        Raises:
            InputFileMissing: If the function definition file does not exist.
            InputFileFormatError: If the JSON file has invalid JSON syntax.
            ModelValidationError: If a function fails schema validation.
        """
        return InputReader.load_functions(
            InputReader.get_full_path(self.args.functions_definition, "input")
        )

    def get_prompts(self) -> List[Prompt]:
        """Loads and returns the list of prompts from the specified input file.

        Returns:
            List[Prompt]: A list of validated `Prompt` objects.

        Raises:
            InputFileMissing: If the prompt input file does not exist.
            InputFileFormatError: If the JSON file has invalid JSON syntax.
            ModelValidationError: If a prompt fails Pydantic validation.
        """
        return InputReader.load_prompts(
            InputReader.get_full_path(self.args.input, "input")
        )

    def get_output_file(self) -> str:
        """Resolves and returns the full path for the output file.

        Returns:
            str: The target output file path.
        """
        return InputReader.get_full_path(self.args.output, "output")

    def error(self, message) -> Never:
        """Overrides ArgumentParser error handler to raise a custom exception.

        Args:
            message (str): The error message explaining why parsing failed.

        Raises:
            CommandLineArgumentsError: Whenever argument parsing fails.
        """
        raise CommandLineArgumentsError(message)

    @staticmethod
    def get_full_path(file_name: str, directory: str) -> str:
        """Constructs a relative path if only a filename is provided.

        If `file_name` contains no directory components, it prefixes it with
        `data/<directory>/`. Otherwise, it returns `file_name` unchanged.

        Args:
            file_name (str): Path or filename to resolve.
            directory (str): Target subfolder in `data/` ('input' or 'output').

        Returns:
            str: Resolved relative or absolute file path.
        """
        if os.path.dirname(file_name) == "":
            return f"data/{directory}/" + file_name
        else:
            return file_name

    @staticmethod
    def load_prompts(file_name: str) -> List[Prompt]:
        """Reads a JSON file and parses its contents into Prompt objects.

        Args:
            file_name (str): Path to the JSON file containing prompt data.

        Returns:
            List[Prompt]: A list of validated `Prompt` instances.

        Raises:
            InputFileMissing: If `file_name` cannot be found.
            InputFileFormatError: If the file contains invalid JSON syntax.
            ModelValidationError: If the JSON data doesn't match `Prompt`.
        """
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
        """Reads a JSON file and parses its contents into Function objects.

        Args:
            file_name (str): Path to the JSON file with function definitions.

        Returns:
            List[Function]: A list of instantiated `Function` objects.

        Raises:
            InputFileMissing: If `file_name` cannot be found.
            InputFileFormatError: If the file contains invalid JSON syntax.
            ModelValidationError: If the parameters fail validation.
        """
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
