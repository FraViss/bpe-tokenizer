"""BPE tokenizer encode and decode using a trained vocabulary and merge rules."""


class BPETokenizer:
    """Byte Pair Encoding tokenizer for text encoding and decoding.

    Applies learned merge rules word-by-word and maps tokens to integer IDs
    using a fixed vocabulary.
    """

    def __init__(self, vocab: list[str], merges: list[tuple[str, str, str]]) -> None:
        """Initialize the tokenizer with a vocabulary and merge rules.

        Args:
            vocab: Sorted list of token strings from training.
            merges: Ordered list of ``(symbol_a, symbol_b, merged)`` tuples.
        """
        self.vocab = vocab
        self.token_to_id = {token: index for index, token in enumerate(vocab)}
        self.id_to_token = {index: token for index, token in enumerate(vocab)}
        self.merges = merges

    def _tokenize_word(self, word: str) -> list[str]:
        """Tokenize a single word by applying BPE merge rules.

        Represents ``word`` as individual characters with ``"</w>"`` appended,
        then applies each merge rule in order. For each rule, repeatedly finds
        the first adjacent ``(a, b)`` pair, merges it, and rescans until no
        more matches remain for that rule.

        Args:
            word: A single whitespace-free word.

        Returns:
            The list of BPE tokens for ``word``.
        """
        symbols = list(word) + ["</w>"]

        for symbol_a, symbol_b, merged in self.merges:
            while True:
                merged_in_pass = False
                for index in range(len(symbols) - 1):
                    if symbols[index] == symbol_a and symbols[index + 1] == symbol_b:
                        symbols = symbols[:index] + [merged] + symbols[index + 2 :]
                        merged_in_pass = True
                        break
                if not merged_in_pass:
                    break

        return symbols

    def encode(self, text: str) -> list[int]:
        """Encode text into a flat list of token IDs.

        Splits ``text`` on whitespace, tokenizes each word, and maps each
        token to its integer ID.

        Args:
            text: Input text to encode.

        Returns:
            A flat list of integer token IDs.

        Raises:
            ValueError: If a produced token is not present in the vocabulary.
        """
        ids: list[int] = []
        for word in text.split():
            for token in self._tokenize_word(word):
                if token not in self.token_to_id:
                    raise ValueError(f"Unknown token: {token!r}")
                ids.append(self.token_to_id[token])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into text.

        Maps each ID to its token string, concatenates the tokens, replaces
        word-boundary markers with spaces, and strips outer whitespace.

        Args:
            ids: List of integer token IDs.

        Returns:
            The reconstructed text string.
        """
        text = "".join(self.id_to_token[token_id] for token_id in ids)
        return text.replace("</w>", " ").strip()
