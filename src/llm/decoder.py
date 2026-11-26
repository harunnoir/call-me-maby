"""Choose and generate valid values from the model output."""

from typing import List, Set

from .language_model import LanguageModel


def choose_best_token(logits: List[float], allowed_tokens: Set[int]) -> int:
    """Select the highest-scoring token from an allowed set.

    Tokens that are not present in allowed_tokens are assigned negative
    infinity so they cannot be selected. The remaining token with the highest
    score is returned.

    Args:
        logits: Score for each possible token.
        allowed_tokens: Token IDs that are allowed to be selected.

    Returns:
        The ID of the highest-scoring allowed token.

    Raises:
        RuntimeError: If no token can be selected.
    """
    masked_logits = [float("-inf")] * len(logits)

    for token_id in allowed_tokens:
        masked_logits[token_id] = logits[token_id]

    best_token = -1
    best_score = float("-inf")
    for token_id, score in enumerate(masked_logits):
        if score > best_score:
            best_score = score
            best_token = token_id

    if best_token == -1:
        raise RuntimeError("No valid token was available during decoding")

    return best_token


def choose_tool(
    model: LanguageModel,
    tool_tokens: List[List[int]],
    context_tokens: List[int],
) -> str:
    """Select one available function name using token-by-token constraints.

    At each step, only token IDs that continue at least one available function
    name are allowed. The highest-scoring allowed token is appended to the
    generated function name until no more valid continuation
    exists.

    Args:
        model: LanguageModel used to calculate token scores and decode tokens.
        tool_tokens: Tokenized function names.
        context_tokens: Token IDs representing the text already given to the
            model. This list is extended while the function name is generated.

    Returns:
        The selected function name as text.
    """
    generated: List[int] = []
    longest_name = max((len(tokens) for tokens in tool_tokens), default=0)

    for _ in range(longest_name):
        logits = model.logits(context_tokens)
        allowed_tokens: Set[int] = set()

        for tokens in tool_tokens:
            if tokens[:len(generated)] == generated:
                if len(tokens) > len(generated):
                    allowed_tokens.add(tokens[len(generated)])

        if not allowed_tokens:
            break

        next_token = choose_best_token(logits, allowed_tokens)
        generated.append(next_token)
        context_tokens.append(next_token)

    return model.decode(generated)


def generate_number(model: LanguageModel, context_tokens: List[int]) -> str:
    """Generate numeric text using the model's allowed number tokens.

    Generation stops when a token from model.number_stops is selected or after
    a maximum of 20 iterations. A valid numeric string without a decimal point
    or exponent is converted to decimal form by appending ".0".

    Args:
        model: LanguageModel used to calculate token scores and decode tokens.
        context_tokens: Token IDs representing the text already given to the
            model. This list is extended while the number is generated.

    Returns:
        A string that can be converted to a Python float. Returns "0.0" when
        the generated text cannot be converted to a float.
    """
    generated = ""

    for _ in range(20):
        logits = model.logits(context_tokens)

        best_token = 0
        best_score = float("-inf")
        for token_id, score in enumerate(logits):
            if score > best_score:
                best_score = score
                best_token = token_id

        if best_token in model.number_stops:
            break

        best_token = choose_best_token(logits, set(model.number_tokens))
        generated += model.decode([best_token])
        context_tokens.append(best_token)

    if generated and "." not in generated and "e" not in generated.lower():
        generated += ".0"

    try:
        float(generated)
    except ValueError:
        return "0.0"
    return generated


def generate_string(model: LanguageModel, context_tokens: List[int]) -> str:
    """Generate string content until a double-quote token is encountered.

    The function generates at most 100 tokens. It checks the token text for a
    double quote and stops when one is found. The generated token IDs are also
    added to context_tokens.

    Args:
        model: LanguageModel used to calculate token scores and decode tokens.
        context_tokens: Token IDs representing the text already given to the
            model. This list is modified during generation.

    Returns:
        The generated string content without the terminating double quote.
    """
    context_tokens.append(model.quote_token)
    generated = ""

    for _ in range(100):
        logits = model.logits(context_tokens)
        next_token = choose_best_token(logits, set(range(len(logits))))
        raw_text = model.token_text.get(next_token, "")

        if '"' in raw_text:
            generated += raw_text[:raw_text.find('"')]
            break

        generated += model.decode([next_token])
        context_tokens.append(next_token)

    context_tokens.append(model.quote_token)
    return generated


def generate_boolean(model: LanguageModel, context_tokens: List[int]) -> bool:
    """Choose between the two JSON boolean values true and false.

    The function compares the scores of the first token of the encoded "true"
    and "false" values, then adds the selected
    token sequence to context_tokens.

    Args:
        model: LanguageModel used to encode the boolean values and calculate
            their scores.
        context_tokens: Token IDs representing the text already given to the
            model. This list is extended with the selected boolean tokens.

    Returns:
        True when "true" receives the higher or equal score; otherwise False.
    """
    true_tokens = model.encode("true")
    false_tokens = model.encode("false")
    logits = model.logits(context_tokens)

    true_score = float("-inf")
    false_score = float("-inf")
    if true_tokens:
        true_score = logits[true_tokens[0]]
    if false_tokens:
        false_score = logits[false_tokens[0]]

    if true_score >= false_score:
        context_tokens.extend(true_tokens)
        return True

    context_tokens.extend(false_tokens)
    return False
