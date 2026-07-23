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


class JSONExtractorTypeError(JSONExtractorError):
    def __init__(self, type: str):
        super().__init__(
            f"Wrong parameter type: {type}."
        )


class JSONExtractorParsingError(JSONExtractorError):
    def __init__(self, param_name: str):
        super().__init__(
            f"Not possible to parse JSON outout for parameter: {param_name}."
        )
