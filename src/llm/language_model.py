"""Small wrapper around the provided language-model SDK."""

import json
from typing import Dict, List, Set, Tuple

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, PrivateAttr


NUMBER_CHARS = frozenset("0123456789-+.eE")
NUMBER_ENDINGS = frozenset({",", ", ", "}}", "}}Ċ"})


class LanguageModel(BaseModel):
    """Wrap the language-model object and expose token-generation operations.

    Attributes:
        model_name: Name passed to Small_LLM_Model when the model is created.
    """

    model_name: str = "Qwen/Qwen3-0.6B"

    _model: Small_LLM_Model = PrivateAttr()
    _token_text: Dict[int, str] = PrivateAttr(default_factory=dict)
    _number_tokens: List[int] = PrivateAttr(default_factory=list)
    _number_stops: Set[int] = PrivateAttr(default_factory=set)
    _comma_tokens: List[int] = PrivateAttr(default_factory=list)
    _parameters_start_tokens: List[int] = PrivateAttr(default_factory=list)
    _quote_token: int = PrivateAttr()

    def model_post_init(self, _context: object) -> None:
        """Load the model and prepare token information used during decoding.

        The method creates Small_LLM_Model, loads its vocabulary, prepares the
        token-text mapping, identifies number-related tokens, and stores token
        sequences needed when generating function-call JSON.

        Args:
            _context: Initialization context supplied to this Pydantic model.

        Returns:
            None.
        """
        self._model = Small_LLM_Model(model_name=self.model_name)
        vocabulary = self._read_vocabulary()
        self._token_text = self._reverse_vocabulary(vocabulary)
        self._number_tokens, self._number_stops = self._find_number_tokens(
            vocabulary
        )
        self._comma_tokens = self.encode(", ")
        self._parameters_start_tokens = self.encode('", "parameters": {')
        self._quote_token = self._required_token(vocabulary, '"')

    def _read_vocabulary(self) -> Dict[str, int]:
        """Read the vocabulary file associated with the language model.

        The vocabulary file is loaded as JSON and must contain a dictionary.
        Keys are converted to strings and token IDs are converted to integers.

        Returns:
            A mapping from token text to token ID.

        Raises:
            RuntimeError: If the vocabulary cannot be read,
            parsed, or converted into the expected dictionary structure.
        """
        path = self._model.get_path_to_vocab_file()

        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read vocabulary: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Vocabulary must contain a JSON object")

        vocabulary: Dict[str, int] = {}
        for text, token_id in data.items():
            vocabulary[str(text)] = int(token_id)
        return vocabulary

    @staticmethod
    def _reverse_vocabulary(vocabulary: Dict[str, int]) -> Dict[int, str]:
        """Reverse a token-text-to-ID vocabulary mapping.

        Args:
            vocabulary: Mapping from token text to token ID.

        Returns:
            A mapping from token ID to token text.
        """
        result: Dict[int, str] = {}
        for text, token_id in vocabulary.items():
            result[token_id] = text
        return result

    @staticmethod
    def _find_number_tokens(
        vocabulary: Dict[str, int],
    ) -> Tuple[List[int], Set[int]]:
        """Find vocabulary tokens that can participate in number generation.

        A token is placed in the number-token list when all
        of its characters are contained in NUMBER_CHARS.
        A token is placed in the number-stop set when
        its stripped text matches one of NUMBER_ENDINGS.

        Args:
            vocabulary: Mapping from token text to token ID.

        Returns:
            A tuple containing the token IDs allowed
            for number generation and the
            token IDs that signal the end of a number.
        """
        number_tokens: List[int] = []
        number_stops: Set[int] = set()

        for text, token_id in vocabulary.items():
            if text and set(text) <= NUMBER_CHARS:
                number_tokens.append(token_id)
            if text.strip() in NUMBER_ENDINGS:
                number_stops.add(token_id)

        return number_tokens, number_stops

    @staticmethod
    def _required_token(vocabulary: Dict[str, int], text: str) -> int:
        """Find the token ID for an exact token string.

        Args:
            vocabulary: Mapping from token text to token ID.
            text: Exact token text to search for.

        Returns:
            The token ID associated with text.

        Raises:
            RuntimeError: If text is not present in the vocabulary.
        """
        token_id = vocabulary.get(text)
        if token_id is None:
            raise RuntimeError(
                f"Required vocabulary token is missing: {text!r}")
        return token_id

    def encode(self, text: str) -> List[int]:
        """Convert text into a list of token IDs.

        Args:
            text: Text to encode using the underlying language model.

        Returns:
            A Python list containing the token IDs produced by the underlying
            model.
        """
        encoded = self._model.encode(text)
        return [int(token_id) for token_id in encoded[0].tolist()]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back into text.

        Args:
            token_ids: Token IDs to decode.

        Returns:
            The text returned by the underlying language model decoder.
        """
        return str(self._model.decode(token_ids))

    def logits(self, token_ids: List[int]) -> List[float]:
        """Get the model's scores for the next token.

        Args:
            token_ids: Token IDs representing the current input context.

        Returns:
            A list of floating-point scores returned by the underlying model.
        """
        scores = self._model.get_logits_from_input_ids(token_ids)
        return [float(score) for score in scores]

    @property
    def token_text(self) -> Dict[int, str]:
        """Return the mapping from token IDs to their text.

        Returns:
            The dictionary stored in _token_text.
        """
        return self._token_text

    @property
    def number_tokens(self) -> List[int]:
        """Return the token IDs allowed during number generation.

        Returns:
            A list of token IDs stored in _number_tokens.
        """
        return self._number_tokens

    @property
    def number_stops(self) -> Set[int]:
        """Return the token IDs that end number generation.

        Returns:
            A set of token IDs stored in _number_stops.
        """
        return self._number_stops

    @property
    def comma_tokens(self) -> List[int]:
        """Return the token IDs representing the separator used between
        parameters.

        Returns:
            A list of token IDs stored in _comma_tokens.
        """
        return self._comma_tokens

    @property
    def parameters_start_tokens(self) -> List[int]:
        """Return the token IDs that start the parameters object.

        Returns:
            A list of token IDs stored in _parameters_start_tokens.
        """
        return self._parameters_start_tokens

    @property
    def quote_token(self) -> int:
        """Return the token ID used for a double quote.

        Returns:
            The token ID stored in _quote_token.
        """
        return self._quote_token
