"""
Download HotpotQA distractor dev set, build the corpus of unique paragraphs,
and sample a frozen eval set + a dev set.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

HOTPOTQA_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)
RAW_PATH = Path("data/raw/hotpotqa_dev.json")
CORPUS_PATH = Path("data/processed/corpus.json")
TEST_QUESTIONS_PATH = Path("data/processed/test_questions.json")
DEV_QUESTIONS_PATH = Path("data/processed/dev_questions.json")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_hotpotqa(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    print(f"Downloading HotpotQA dev set → {dest} ...")
    resp = requests.get(HOTPOTQA_URL, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Saved {dest.stat().st_size / 1e6:.1f} MB")


def build_corpus(raw_data: list[dict]) -> list[dict]:
    """Extract unique paragraphs from the distractor context field."""
    seen: set[str] = set()
    docs: list[dict] = []
    for item in tqdm(raw_data, desc="Building corpus"):
        for title, sentences in item["context"]:
            text = " ".join(sentences).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            docs.append({"doc_id": len(docs), "title": title, "text": text})
    return docs


def build_questions(raw_data: list[dict]) -> list[dict]:
    """Flatten each QA item into a standardised format."""
    questions = []
    for item in raw_data:
        supporting_titles = {title for title, _ in item.get("supporting_facts", [])}
        questions.append(
            {
                "question_id": item["_id"],
                "question": item["question"],
                "answer": item["answer"],
                "supporting_titles": list(supporting_titles),
                "type": item.get("type", ""),
                "level": item.get("level", ""),
            }
        )
    return questions


def main(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    ds_cfg = cfg["dataset"]
    seed: int = ds_cfg["seed"]
    test_size: int = ds_cfg["test_size"]
    dev_size: int = ds_cfg["dev_size"]

    download_hotpotqa(RAW_PATH)

    print("Loading raw data...")
    with RAW_PATH.open() as f:
        raw = json.load(f)
    print(f"Loaded {len(raw):,} examples")

    # Build corpus
    print("Extracting corpus paragraphs...")
    corpus = build_corpus(raw)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w") as f:
        json.dump(corpus, f)
    print(f"Corpus: {len(corpus):,} unique paragraphs → {CORPUS_PATH}")

    # Sample questions
    questions = build_questions(raw)
    rng = random.Random(seed)
    rng.shuffle(questions)

    test_q = questions[:test_size]
    dev_q = questions[test_size: test_size + dev_size]

    with TEST_QUESTIONS_PATH.open("w") as f:
        json.dump(test_q, f, indent=2)
    with DEV_QUESTIONS_PATH.open("w") as f:
        json.dump(dev_q, f, indent=2)

    print(f"Test set : {len(test_q):,} questions → {TEST_QUESTIONS_PATH}")
    print(f"Dev set  : {len(dev_q):,} questions → {DEV_QUESTIONS_PATH}")
    print("Dataset preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
