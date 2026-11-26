"""Define the Pydantic models used by the application."""

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field


ValueTypeName = Literal["string", "number", "integer", "boolean"]


class ValueType(BaseModel):
    """Describe the allowed type of one function value.

    Attributes:
        type: The value type. The accepted values are "string", "number",
            "integer", and "boolean".
    """

    model_config = ConfigDict(extra="forbid")
    type: ValueTypeName


class ToolDefinition(BaseModel):
    """Describe one function that the model is allowed to select.

    Attributes:
        name: Name of the function.
        description: Human-readable description of what the function does.
        parameters: Mapping of parameter names to their expected value types.
        returns: Type of value returned by the function.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: Dict[str, ValueType]
    returns: ValueType


class PromptRecord(BaseModel):
    """Store one user prompt read from the input data.

    Attributes:
        prompt: The natural-language request that will be given to the
            function-calling system.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=1)


class CallRecord(BaseModel):
    """Validate one generated function-call result.

    Attributes:
        prompt: The original user request.
        name: Name of the function selected for the request.
        parameters: Arguments generated for the selected function.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    parameters: dict[str, Any]
