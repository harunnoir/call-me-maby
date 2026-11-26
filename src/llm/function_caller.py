"""Turn a user prompt into a function call."""

import json
from typing import Any, List

from pydantic import BaseModel, ConfigDict

from ..models import ToolDefinition
from ..prompting import build_instruction
from .decoder import (
    choose_tool,
    generate_boolean,
    generate_number,
    generate_string,
)
from .language_model import LanguageModel


class FunctionCaller(BaseModel):
    """Generate function calls from natural-language prompts.

    Attributes:
        model: LanguageModel used to select functions and generate arguments.
        tools: Function definitions that are available for selection.
    """

    model_config = ConfigDict(extra="forbid")
    model: LanguageModel
    tools: List[ToolDefinition]

    def generate(self, prompt: str) -> dict[str, Any]:
        """Generate one function-call result for a user prompt.

        The method builds the model instruction, constrains the selection
        to the available function names, finds the selected ToolDefinition, and
        generates each parameter according to its declared type.

        Args:
            prompt: Natural-language request that the model should interpret.

        Returns:
            A dictionary containing "prompt", "name", and "parameters".
        """
        instruction = build_instruction(self.tools)
        tool_tokens = [self.model.encode(tool.name) for tool in self.tools]
        prefix = '{"prompt": ' + json.dumps(prompt) + ', "name": "'
        context = instruction + prompt + prefix
        context_tokens = self.model.encode(context)

        tool_name = choose_tool(self.model, tool_tokens, context_tokens)
        tool = self._find_tool(tool_name)

        parameters: dict[str, Any] = {}
        result = {
            "prompt": prompt,
            "name": tool.name,
            "parameters": parameters,
        }

        context_tokens.extend(self.model.parameters_start_tokens)
        parameter_names = list(tool.parameters.keys())

        for index, parameter_name in enumerate(parameter_names):
            key_text = f'"{parameter_name}": '
            context_tokens.extend(self.model.encode(key_text))
            parameters[parameter_name] = self._generate_value(
                tool.parameters[parameter_name].type,
                context_tokens,
            )

            if index < len(parameter_names) - 1:
                context_tokens.extend(self.model.comma_tokens)

        return result

    def _find_tool(self, name: str) -> ToolDefinition:
        """Find the function definition with the given name.

        Args:
            name: Function name selected by the model.

        Returns:
            The ToolDefinition whose name matches name.

        Raises:
            ValueError: If no available ToolDefinition has the requested name.
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise ValueError(f"LLM selected an unmapped function: '{name}'")

    def _generate_value(
        self,
        value_type: str,
        context_tokens: List[int],
    ) -> Any:
        """Generate one function argument according to its declared type.

        String values are generated with generate_string(), boolean values with
        generate_boolean(), and numeric values with generate_number().

        Args:
            value_type: Declared parameter type.
            context_tokens: Token IDs representing the current
            generation context. This list is modified
            while the value is generated.

        Returns:
            A generated string, boolean, integer, or floating-point number.
        """
        if value_type == "string":
            return generate_string(self.model, context_tokens)
        if value_type == "boolean":
            return generate_boolean(self.model, context_tokens)

        number = generate_number(self.model, context_tokens)
        if value_type == "integer":
            return int(float(number))
        return float(number)
