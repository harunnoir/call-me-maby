"""Build the instructions sent to the language model."""

import json
from typing import List

from .models import ToolDefinition


def build_instruction(tools: List[ToolDefinition]) -> str:
    """Build the text instruction given to the language model.

    The instruction contains the available function names, descriptions,
    parameter types, and additional rules used during generation.

    Args:
        tools: Function definitions that the model may choose from.

    Returns:
        A string containing the model instructions and serialized function
        definitions.
    """
    function_definitions = []

    for tool in tools:
        parameters = {}
        for name, value_type in tool.parameters.items():
            parameters[name] = {"type": value_type.type}

        function_definitions.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            }
        )

    return (
        "You are a helpful assistant that calls functions.\n"
        "Available functions: "
        f"{json.dumps(function_definitions, ensure_ascii=False)}\n"
        "Rules:\n"
        "- choose the most appropriate function.\n"
        "- Extract arguments precisely from the user request.\n"
        "- for regex patterns, use standard regex syntax like [0-9]+ "
        "for numbers and [aeiouAEIOU] for vowels.\n"
        "- asterisks replacement should be \"*\" only, not the word "
        "asterisks.\n"
        "Return only a function call with prompt, name, and parameters.\n"
    )
