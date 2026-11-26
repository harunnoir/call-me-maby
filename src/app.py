"""Run the program from start to finish."""

import json
import sys
import time
from typing import Any, List

from pydantic import ValidationError

from .cli import read_cli
from .files import read_prompts, read_tools, write_results
from .models import CallRecord
from .llm.function_caller import FunctionCaller
from .llm.language_model import LanguageModel


def main() -> None:
    """Run the complete function-calling application.

    The function reads the input and function-definition files, creates the
    language model and function caller, processes every user prompt, validates
    each generated result, and writes the successful results to the output
    file.

    Returns:
        None. The function writes results to the output file and may terminate
        the program with a non-zero exit code when an error occurs.
    """
    started = time.perf_counter()
    args = read_cli()

    print("Loading input files...")
    tools = read_tools(args.functions_definition)
    prompts = read_prompts(args.input)
    if not tools or not prompts:
        print("Error: required input data could not be loaded.",
              file=sys.stderr)
        sys.exit(1)

    print("Loading model...")
    try:
        caller = FunctionCaller(model=LanguageModel(), tools=tools)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Initialization error: {exc}", file=sys.stderr)
        sys.exit(1)

    results: List[dict[str, Any]] = []
    failures = 0

    print(f"Processing {len(prompts)} prompts...")
    for index, item in enumerate(prompts, 1):
        print(f"[{index}/{len(prompts)}] {item.prompt}")
        prompt_started = time.perf_counter()

        try:
            raw_result = caller.generate(item.prompt)
            result = CallRecord(**raw_result)
            elapsed = time.perf_counter() - prompt_started
            print(
                f"  -> {result.name} "
                f"{json.dumps(result.parameters, ensure_ascii=False)} "
                f"({elapsed:.2f}s)"
            )
            results.append(result.model_dump())
        except (ValidationError, RuntimeError, ValueError) as exc:
            print(f"  -> Generation failed: {exc}")
            failures += 1

    try:
        write_results(args.output, results)
    except (OSError, TypeError) as exc:
        print(f"Error saving output file: {exc}", file=sys.stderr)
        sys.exit(1)

    total = time.perf_counter() - started
    print(f"Results saved to: {args.output}")
    print(f"Successful: {len(results)} | Failed: {failures}")
    print(f"Total time: {total:.2f}s")

    if failures:
        sys.exit(1)
