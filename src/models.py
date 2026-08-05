from typing import List, Dict, Any, Self
from pydantic import BaseModel, model_validator, ValidationError
from pydantic_core import PydanticCustomError


class Prompt(BaseModel):
    """Pydantic model representing a user input prompt.

    Attributes:
        prompt (str): Raw prompt text.
    """

    prompt: str

    @model_validator(mode='after')
    def check_empty_prompt(self) -> Self:
        """Validates that the prompt string is not empty.

        Returns:
            Self: Validated model instance.

        Raises:
            ValidationError: If prompt string is empty.
        """
        if self.prompt == "":
            raise ValidationError.from_exception_data(
                title="Prompt",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_empty_prompt",
                        "Prompt is empty"
                    ),
                    "loc": ("prompt",),
                    "input": {
                        "prompt": self.prompt
                    }
                }]
            )
        return self


class Returns(BaseModel):
    """Pydantic model representing return type specifications.

    Attributes:
        type (str): Return type name.
    """

    type: str

    @model_validator(mode='after')
    def check_wrong_type(self) -> Self:
        """Validates that return type is an allowed primitive type.

        Returns:
            Self: Validated model instance.

        Raises:
            ValidationError: If return type is invalid.
        """
        if self.type not in ["number", "integer", "string", "boolean"]:
            raise ValidationError.from_exception_data(
                title="Returns",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_wrong_type",
                        "Wrong returns type"
                    ),
                    "loc": ("type",),
                    "input": {
                        "returns": self.type
                    }
                }]
            )
        return self


class Parameter(BaseModel):
    """Pydantic model representing a function parameter schema.

    Attributes:
        name (str): Parameter identifier.
        type (str): Primitive parameter type name.
    """

    name: str
    type: str

    @model_validator(mode='after')
    def check_empty_param_name(self) -> Self:
        """Validates that the parameter name is not empty.

        Returns:
            Self: Validated model instance.

        Raises:
            ValidationError: If parameter name is empty.
        """
        if self.name == "":
            raise ValidationError.from_exception_data(
                title="Name",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_empty_param_name",
                        "Parameter name is empty"
                    ),
                    "loc": ("name",),
                    "input": {
                        "name": self.name
                    }
                }]
            )
        return self

    @model_validator(mode='after')
    def check_wrong_type(self) -> Self:
        """Validates that parameter type is supported.

        Returns:
            Self: Validated model instance.

        Raises:
            ValidationError: If parameter type is invalid.
        """
        if self.type not in ["number", "integer", "string", "boolean"]:
            raise ValidationError.from_exception_data(
                title="Type",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_wrong_type",
                        "Wrong type"
                    ),
                    "loc": ("type",),
                    "input": {
                        "type": self.type
                    }
                }]
            )
        return self


class Function(BaseModel):
    """Pydantic model representing a supported function definition.

    Attributes:
        name (str): Function name.
        description (str): Functional purpose description.
        parameters (List[Parameter]): Parameter specifications list.
        returns (Returns): Expected return type container.
    """

    name: str
    description: str
    parameters: List[Parameter]
    returns: Returns

    @model_validator(mode='before')
    @classmethod
    def reshape_parameters(cls, data: Any) -> Any:
        """Reshapes the JSON's {name: {type: ...}} parameter.

        Returns:
            data: List `Parameter` expected or None if input had wrong type.
        """
        if isinstance(data, dict) and isinstance(data.get("parameters"), dict):
            data = {
                **data,
                "parameters": [
                    {"name": name, **p} if isinstance(p, dict) else p
                    for name, p in data["parameters"].items()
                ],
            }
        return data

    @model_validator(mode='after')
    def check_empty_func_name(self) -> Self:
        """Validates that the function name is not empty.

        Returns:
            Self: Validated model instance.

        Raises:
            ValidationError: If function name is empty.
        """
        if self.name == "":
            raise ValidationError.from_exception_data(
                title="Name",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_empty_func_name",
                        "Function name is empty"
                    ),
                    "loc": ("name",),
                    "input": {
                        "name": self.name
                    }
                }]
            )
        return self

    @model_validator(mode='after')
    def check_empty_func_desc(self) -> Self:
        """Validates that the function description is not empty.

        Returns:
            Self: Validated model instance.

        Raises:
            ValidationError: If description string is empty.
        """
        if self.description == "":
            raise ValidationError.from_exception_data(
                title="Description",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_empty_func_desc",
                        "Function description is empty"
                    ),
                    "loc": ("description",),
                    "input": {
                        "description": self.description
                    }
                }]
            )
        return self

    @model_validator(mode='after')
    def check_empty_parameters(self) -> Self:
        """Validates that the list of parameters is not empty.

        Returns:
            Self: Validated model instance.

        Raises:
            ValidationError: If parameters list is empty.
        """
        if len(self.parameters) == 0:
            raise ValidationError.from_exception_data(
                title="Parameters",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_empty_parameters",
                        "Parameters list is empty"
                    ),
                    "loc": ("parameters",),
                    "input": {
                        "parameters": self.parameters
                    }
                }]
            )
        return self


class PromptResponse(BaseModel):
    """Pydantic model representing the extraction output response.

    Attributes:
        prompt (str): Original input prompt string.
        function (Function): Selected target function.
        parameters (Dict[str, Any]): Extracted key-value parameters.
    """

    prompt: str
    function: Function
    parameters: Dict[str, Any]

    @model_validator(mode='after')
    def check_parameters(self) -> Self:
        """Validates that extracted parameter keys match function schema.

        Returns:
            Self: Validated model instance.

        Raises:
            ValidationError: If extracted parameters deviate from schema.
        """
        original_parameters = [p.name for p in self.function.parameters]
        output_parameters = [p for p in self.parameters.keys()]
        if original_parameters != output_parameters:
            raise ValidationError.from_exception_data(
                title="Parameters",
                line_errors=[{
                    "type": PydanticCustomError(
                        "check_parameters",
                        "Output parameters don't match original ones"
                    ),
                    "loc": ("name",),
                    "input": {
                        "parameters": output_parameters
                    }
                }]
            )
        return self
