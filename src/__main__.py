import argparse
import os
import sys
from .input_files_parser import InputParser, FunctionDefinitionParser
from .errors import InputFileError
from .pipeline import GenerationPipeline


class SilentParser(argparse.ArgumentParser):  # Suppress argparse's own output by overriding error()
    def error(self, message) -> None:
        raise SystemExit(message)


def get_full_path(file_name: str) -> str:
    if os.path.dirname(file_name) == "":
        return "data/input/" + file_name
    else:
        return file_name


if __name__ == "__main__":
    try:
        parser = SilentParser()
        parser.add_argument("--functions_definition", required=True)
        parser.add_argument("--input", required=True)
        # parser.add_argument("--output", required=True)
        args = parser.parse_args()

        functions_file = get_full_path(args.functions_definition)
        functions = FunctionDefinitionParser().load(functions_file)
        prompts = InputParser().load(get_full_path(args.input))
        pipeline = GenerationPipeline(functions)
        for prompt in prompts:
            response = pipeline.generate_prompt_response(prompt)
            print(response)
            break
    except SystemExit as e:
        print(f"Error: Missing required argument '--{str(e).split("--")[1]}'")
        sys.exit(1)  # Not strictly needed, but without it the program would continue executing after the except block
    except InputFileError as e:
        print(f"Error: {e}")

# This file is needed tp run python -m src
# __init__.py won't work for this. It's executed when the package is imported, not when it's run with -m. It has no if __name__ == "__main__" context.

# So python -m src will still look specifically for __main__.py — __init__.py is ignored for that purpose.

# You can have both though:

# __init__.py — runs on import, used to expose the package's public API
# __main__.py — runs when you do python -m src

        # functions = FunctionDefinitionParser().load(get_full_path(
        #     args.functions_definition
        # ))

        # for function in functions:
        #     print(f"name: {function.name}")
        #     print(f"description: {function.description}")
        #     print(f"parameters: {function.parameters}")
        #     print(f"returns: {function.returns}")
        # for prompt in prompts:
        #     print(prompt.prompt)
