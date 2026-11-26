"""Read data from JSON files and save the results."""

import json
from typing import Any, List, TypeVar

from pydantic import BaseModel, ValidationError

from .models import PromptRecord, ToolDefinition

ModelType = TypeVar("ModelType", bound=BaseModel)


def read_json_array(path: str) -> List[Any]:
    """Read a JSON file whose top-level value must be a list.

    Args:
        path: Path to the JSON file to read.

    Returns:
        The decoded list when the file contains valid JSON with a list at the
        top level. Returns an empty list when the file cannot be read, contains
        invalid JSON, or does not contain a list.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return []
    except PermissionError:
        print(f"Error: permission denied: {path}")
        return []
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}: {exc}")
        return []
    except OSError as exc:
        print(f"Error: could not read {path}: {exc}")
        return []

    if not isinstance(data, list):
        print(f"Error: expected a JSON array in {path}")
        return []
    return data


def validate_items(data: List[Any], model: type[ModelType]) -> List[ModelType]:
    """Validate dictionary items and convert valid items into model objects.

    Each item must be a dictionary that can be accepted by the supplied
    Pydantic model. Invalid items are reported and skipped instead of stopping
    the validation of the remaining items.

    Args:
        data: Values read from an input JSON array.
        model: Pydantic model class used to validate each dictionary.

    Returns:
        A list containing only the items that were successfully validated.
    """
    result: List[ModelType] = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Validation failed at index {index}: expected an object")
            continue
        try:
            result.append(model(**item))
        except ValidationError as exc:
            label = item.get("name") or item.get("prompt") or str(index)
            print(f"Validation failed for {label}: {exc}")

    return result


def read_tools(path: str) -> List[ToolDefinition]:
    """Read and validate function definitions from a JSON file.

    Args:
        path: Path to the JSON file containing function definitions.

    Returns:
        A list of valid ToolDefinition objects. Invalid entries or unreadable
        input result in an empty or partially filled list.
    """
    return validate_items(read_json_array(path), ToolDefinition)


def read_prompts(path: str) -> List[PromptRecord]:
    """Read and validate user prompts from a JSON file.

    Args:
        path: Path to the JSON file containing prompt objects.

    Returns:
        A list of valid PromptRecord objects. Invalid entries or unreadable
        input result in an empty or partially filled list.
    """
    return validate_items(read_json_array(path), PromptRecord)


def write_results(path: str, results: List[dict[str, Any]]) -> None:
    """Write function-call results to a JSON file.

    The destination directory is created when it does not already exist.

    Args:
        path: Path of the JSON file to create or replace.
        results: Results that will be written as a JSON array.

    Returns:
        None.
    """
    import os

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
