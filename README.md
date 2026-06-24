# BPE Tokenizer From Scratch

A Byte Pair Encoding (BPE) tokenizer implemented from scratch in Python, trained on the Tiny Shakespeare corpus. BPE is the subword tokenization algorithm used by GPT-2, GPT-4, and most modern large language models.

---

## Project Structure

```
bpe-tokenizer/
├── bpe/
│   ├── corpus.py       # Corpus download and loading
│   ├── trainer.py      # BPE training algorithm
│   └── tokenizer.py    # Encoding and decoding
├── notebooks/
│   └── demo.ipynb      # Interactive demo with visualizations
├── data/               # Downloaded corpus (not versioned)
├── outputs/            # Generated plots (not versioned)
├── main.py             # CLI entry point
└── requirements.txt
```

---

## How BPE Works

BPE builds a vocabulary iteratively:

1. **Initialize** — start from all unique characters in the corpus
2. **Count pairs** — find the most frequent adjacent symbol pair
3. **Merge** — fuse the pair into a new token and add it to the vocabulary
4. **Repeat** — iterate for N steps to reach the desired vocabulary size

After training, any text can be encoded by applying the learned merge rules in order,
and decoded by reversing the process — with perfect roundtrip fidelity.

```
"hello world"
→ character level:  ['h', 'e', 'l', 'l', 'o', '</w>', 'w', 'o', 'r', 'l', 'd', '</w>']
→ after BPE merges: ['hello</w>', 'world</w>']
→ token IDs:        [431, 289]
```

---

## Quickstart

```bash
git clone https://github.com/FraViss/bpe-tokenizer.git
cd bpe-tokenizer
uv venv
uv pip install -r requirements.txt
python main.py
```

Expected output:

```
Vocabulary size: 564
Sample: 'To be or not to be'
Token IDs: [...]
Decoded: 'To be or not to be'

First 30 vocabulary tokens:
  '\n'
  ' '
  '!'
  ...

First 10 merge rules:
  ('t', 'h') -> 'th'
  ('e', '</w>') -> 'e</w>'
  ...
```

---

## Interactive Demo

The notebook `notebooks/demo.ipynb` walks through the full BPE pipeline step by step,
combining narrative explanations with visualizations. It is pre-executed and renders
directly on GitHub — no local setup required to read it.

To run it locally:

```bash
jupyter notebook notebooks/demo.ipynb
```

| # | Section | What you will see |
|---|---|---|
| 1 | Corpus | Total characters, words, and unique symbols in Tiny Shakespeare |
| 2 | Character Frequencies | Horizontal bar chart of the top 30 most frequent characters |
| 3 | BPE Training | Merge rules learned, vocabulary size after 500 iterations |
| 4 | Vocabulary Growth | Line chart of vocabulary size over merge steps |
| 5 | Top Merge Rules | First 20 merges ranked by order of discovery |
| 6 | Tokenization in Action | Encode and decode real Shakespeare quotes with roundtrip assert |
| 7 | Sequence Compression | Grouped bar chart comparing character count vs BPE token count |

---

## Key Concepts

**Subword tokenization** — BPE operates between character level and word level.
Common words become single tokens; rare words are split into known subwords.
No word is ever out-of-vocabulary.

**Merge rules** — the output of training is an ordered list of rules `(a, b) → ab`.
Encoding applies them in the same order they were learned.

**Roundtrip fidelity** — decoding always reconstructs the original text exactly,
because `</w>` markers preserve word boundaries.

---

## Requirements

- Python 3.12+
- NumPy
- Matplotlib
- Seaborn
- Jupyter
- Requests
- tqdm

---

## Reference

Sennrich, R., Haddow, B., & Birch, A. (2016).
*Neural Machine Translation of Rare Words with Subword Units*. ACL 2016.
https://arxiv.org/abs/1508.07909

---

## License

MIT
