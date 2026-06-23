"""Download and load text corpora for BPE tokenizer training."""

import os
from pathlib import Path

import requests
from tqdm import tqdm

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"

DEFAULT_CORPUS_PATH = DATA_PATH / "shakespeare.txt"


def download_corpus(force: bool = False) -> Path:
    """Download the Tiny Shakespeare corpus and save it locally.

    Fetches the raw text from ``SHAKESPEARE_URL`` and writes it to
    ``data/shakespeare.txt``. If the file already exists and ``force`` is
    ``False``, the download is skipped.

    Args:
        force: Re-download even when the file already exists.

    Returns:
        Path to the saved corpus file.
    """
    os.makedirs(DATA_PATH, exist_ok=True)

    if DEFAULT_CORPUS_PATH.exists() and not force:
        return DEFAULT_CORPUS_PATH

    response = requests.get(SHAKESPEARE_URL, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(DEFAULT_CORPUS_PATH, "wb") as file:
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading corpus") as progress:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))

    return DEFAULT_CORPUS_PATH


def load_corpus(path: Path | str | None = None) -> str:
    """Read a corpus file and return its contents as a single string.

    Args:
        path: Path to the corpus file. Defaults to ``data/shakespeare.txt``.

    Returns:
        The full text of the corpus.
    """
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    return corpus_path.read_text(encoding="utf-8")
