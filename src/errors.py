class CommandLineArgumentsError(Exception):
    """Raised when required command-line arguments are missing."""

    def __init__(self, message: str):
        """Initializes CommandLineArgumentsError with missing argument message.

        Args:
            message (str): Argument error details containing the missing flag.
        """
        super().__init__(
            f"Missing required argument '--{str(message).split("--")[1]}'."
        )


class InputFileError(Exception):
    """Base exception for all input file processing errors."""

    pass


class InputFileMissing(InputFileError):
    """Raised when an expected input file cannot be found."""

    def __init__(self, file: str):
        """Initializes InputFileMissing with the missing file path.

        Args:
            file (str): Path to the missing file.
        """
        super().__init__(f"Wrong input file: {file} doesn't exist.")


class InputFileFormatError(InputFileError):
    """Raised when an input file contains invalid formatting or syntax."""

    def __init__(self, message: str, line: int):
        """Initializes InputFileFormatError with syntax details and line.

        Args:
            message (str): Description of the formatting error.
            line (str): Line number or content where the error occurred.
        """
        super().__init__(
            f"Wrong format in input file: {message} in line {line}."
        )


class ModelValidationError(InputFileError):
    """Raised when input data fails schema or model validation."""

    def __init__(self, error: str):
        """Initializes ModelValidationError with error message.

        Args:
            error (str): Description of the validation failure.
        """
        super().__init__(
            f"Wrong input: {error}."
        )


class JSONExtractorError(Exception):
    """Base exception for all JSON extraction failures."""

    pass


class JSONExtractorTypeError(JSONExtractorError):
    """Raised when an extracted parameter has an unsupported type."""

    def __init__(self, type: str):
        """Initializes JSONExtractorTypeError with the invalid type name.

        Args:
            type (str): The invalid parameter type encountered.
        """
        super().__init__(
            f"Wrong parameter type: {type}."
        )


class JSONExtractorParsingError(JSONExtractorError):
    """Raised when JSON output cannot be parsed for a specific parameter."""

    def __init__(self, param_name: str):
        """Initializes JSONExtractorParsingError with the parameter name.

        Args:
            param_name (str): Name of the parameter that failed parsing.
        """
        super().__init__(
            f"Not possible to parse JSON output for parameter: {param_name}."
        )


class JSONExtractorTimeoutError(JSONExtractorError):
    """Raised when JSON extraction times out during processing."""

    def __init__(self, output: str):
        """Initializes JSONExtractorTimeoutError with partial or raw output.

        Args:
            output (str): Output received before the timeout occurred.
        """
        super().__init__(
            f"Timeout error during extraction: {output}."
        )


class JSONExtractorMissingParamError(JSONExtractorError):
    """Raised when required parameters are missing from the prompt output."""

    def __init__(self, prompt: str):
        """Initializes JSONExtractorMissingParamError with the prompt context.

        Args:
            prompt (str): Prompt or content missing required parameters.
        """
        super().__init__(
            f"Missing parameters in: {prompt}."
        )
