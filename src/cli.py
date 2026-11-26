"""Read the options given when the program is started."""

import argparse


DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def read_cli() -> argparse.Namespace:
    """Read the file paths provided on the command line.

    The supported arguments are the function-definition file, the input
    prompt file, and the output file. Each argument has a default path defined
    by DEFAULT_FUNCTIONS, DEFAULT_INPUT, and DEFAULT_OUTPUT.

    Returns:
        An argparse.Namespace containing the selected file paths.
    """
    parser = argparse.ArgumentParser(description="LLM function caller")
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()
