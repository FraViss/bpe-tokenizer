"""BPE training: vocabulary initialization, pair statistics, and merge rules."""

from collections import Counter

from tqdm import tqdm


def get_vocab(text: str) -> Counter:
    """Build a word-level vocabulary from raw text.

    Splits ``text`` on whitespace and represents each word as a tuple of
    characters with ``"</w>"`` appended as the final symbol.

    Args:
        text: Raw corpus text.

    Returns:
        A Counter mapping each character-tuple to its frequency.
    """
    vocab: Counter[tuple[str, ...]] = Counter()
    for word in text.split():
        word_tuple = tuple(word) + ("</w>",)
        vocab[word_tuple] += 1
    return vocab


def get_pairs(vocab: Counter) -> Counter:
    """Count adjacent symbol pairs across all words in the vocabulary.

    Args:
        vocab: A Counter mapping word-tuples to frequencies.

    Returns:
        A Counter mapping each adjacent pair to its total frequency.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for word_tuple, frequency in vocab.items():
        for index in range(len(word_tuple) - 1):
            pair = (word_tuple[index], word_tuple[index + 1])
            pairs[pair] += frequency
    return pairs


def merge_vocab(pair: tuple[str, str], vocab: Counter) -> Counter:
    """Merge a symbol pair into a single token across the vocabulary.

    Replaces every non-overlapping occurrence of adjacent ``(a, b)`` in each
    word-tuple with the merged symbol ``a + b``. The input vocabulary is not
    modified in place.

    Args:
        pair: The adjacent symbol pair to merge.
        vocab: The current vocabulary Counter.

    Returns:
        A new Counter with the merge applied.
    """
    symbol_a, symbol_b = pair
    merged = symbol_a + symbol_b
    new_vocab: Counter[tuple[str, ...]] = Counter()

    for word_tuple, frequency in vocab.items():
        merged_word: list[str] = []
        index = 0
        while index < len(word_tuple):
            if (
                index < len(word_tuple) - 1
                and word_tuple[index] == symbol_a
                and word_tuple[index + 1] == symbol_b
            ):
                merged_word.append(merged)
                index += 2
            else:
                merged_word.append(word_tuple[index])
                index += 1
        new_vocab[tuple(merged_word)] = frequency

    return new_vocab


def train(text: str, num_merges: int) -> dict:
    """Train a BPE tokenizer on ``text`` for a fixed number of merges.

    Repeatedly finds the most frequent adjacent pair, merges it into a new
    token, and records the merge rule until ``num_merges`` iterations are
    completed or no pairs remain.

    Args:
        text: Raw corpus text.
        num_merges: Maximum number of merge operations to perform.

    Returns:
        A dict with keys ``"vocab"`` (sorted token list), ``"merges"`` (ordered
        merge rules), and ``"vocab_size"`` (total token count).
    """
    vocab = get_vocab(text)

    token_set: set[str] = set()
    for word_tuple in vocab:
        token_set.update(word_tuple)

    merges: list[tuple[str, str, str]] = []

    for _ in tqdm(range(num_merges), desc="Training BPE"):
        pairs = get_pairs(vocab)
        if not pairs:
            break

        best_pair, _ = pairs.most_common(1)[0]
        symbol_a, symbol_b = best_pair
        merged = symbol_a + symbol_b

        token_set.add(merged)
        vocab = merge_vocab(best_pair, vocab)
        merges.append((symbol_a, symbol_b, merged))

    return {
        "vocab": sorted(token_set),
        "merges": merges,
        "vocab_size": len(token_set),
    }
