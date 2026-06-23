"""CLI entry point for training and demonstrating the BPE tokenizer.

Downloads the Tiny Shakespeare corpus, trains a tokenizer, and runs a
short encode/decode demo suitable for portfolio use.
"""

from bpe.corpus import download_corpus, load_corpus
from bpe.tokenizer import BPETokenizer
from bpe.trainer import train


def main() -> None:
    # Step 1 — Corpus: download (if needed) and load the training text.
    download_corpus()
    text = load_corpus()

    # Step 2 — Training: learn merge rules and build the vocabulary.
    result = train(text, num_merges=500)
    print(f"Vocabulary size: {result['vocab_size']}")

    # Step 3 — Tokenization demo: encode a sample sentence and decode it back.
    tokenizer = BPETokenizer(result["vocab"], result["merges"])

    sample = "To be or not to be"
    token_ids = tokenizer.encode(sample)
    reconstructed = tokenizer.decode(token_ids)

    print(f"Sample: {sample!r}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {reconstructed!r}")

    # Step 4 — Pretty print: show a slice of the learned vocabulary and merges.
    print("\nFirst 30 vocabulary tokens:")
    for token in result["vocab"][:30]:
        print(f"  {token!r}")

    print("\nFirst 10 merge rules:")
    for symbol_a, symbol_b, merged in result["merges"][:10]:
        print(f"  ({symbol_a!r}, {symbol_b!r}) -> {merged!r}")


if __name__ == "__main__":
    main()
