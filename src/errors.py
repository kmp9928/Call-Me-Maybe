class InputFileError(Exception):
    pass


class InputFileMissing(InputFileError):
    def __init__(self, file: str):
        super().__init__(f"Wrong input file: {file} doesn't exist.")


class InputFileFormatError(InputFileError):
    def __init__(self, message: str, line: str):
        super().__init__(
            f"Wrong format in input file: {message} in line {line}."
        )


class ModelValidationError(InputFileError):
    def __init__(self, error: str):
        super().__init__(
            f"Wrong input: {error}."
        )


class JSONExtractorError(Exception):
    pass
