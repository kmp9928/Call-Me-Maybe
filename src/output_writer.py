import json
import os
from typing import List
from .models import PromptResponse


class OutputWriter:
    @staticmethod
    def write_output(file_name: str, responses: List[PromptResponse]) -> None:
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
        directory = os.path.dirname(file_name)

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
