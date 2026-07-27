import json
import os
from typing import List
from .models import PromptResponse


class OutputWriter:
    """Handles writing processed prompt responses to disk in JSON format."""

    @staticmethod
    def write_output(file_name: str, responses: List[PromptResponse]) -> None:
        """Writes prompt response objects to a JSON file.

        Args:
            file_name (str): Destination file path for the output.
            responses (List[PromptResponse]): List of response objects
                to serialize.
        """
        OutputWriter.check_directory(file_name)

        try:
            with open(file_name, "w") as file:
                json.dump(
                    [
                        {
                            "prompt": r.prompt,
                            "name": r.function.name,
                            "parameters": r.parameters
                        }
                        for r in responses
                    ],
                    file,
                    indent=2
                )

        except json.JSONDecodeError:
            print("Error in output.")

    @staticmethod
    def check_directory(file_name: str) -> None:
        """Ensures the target directory exists, creating it if needed.

        Args:
            file_name (str): Target output file path.
        """
        directory = os.path.dirname(file_name)

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
