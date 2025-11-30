import json
from pathlib import Path
from typing import List, Dict, Any


def load_track_a(file_path: str) -> List[Dict[str, Any]]:
    """Load the Track A JSONL dataset.

    Each line in the JSONL file is expected to be a JSON object with the
    following keys (as used in the SemEval task):
        - ``anchor``: the anchor story (string)
        - ``choice_a``: first candidate story (string)
        - ``choice_b``: second candidate story (string)
        - ``label``: ``A`` if ``choice_a`` is more similar, ``B`` otherwise

    The function returns a list of dictionaries with the same keys.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Track A data file not found: {file_path}")

    data = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Map keys from SemEval format to our internal format
                # Expected in file: anchor_text, text_a, text_b, text_a_is_closer
                if "anchor_text" in entry:
                    entry["anchor"] = entry.pop("anchor_text")
                if "text_a" in entry:
                    entry["choice_a"] = entry.pop("text_a")
                if "text_b" in entry:
                    entry["choice_b"] = entry.pop("text_b")
                if "text_a_is_closer" in entry:
                    is_a = entry.pop("text_a_is_closer")
                    entry["label"] = "A" if is_a else "B"

                # Basic validation
                required = {"anchor", "choice_a", "choice_b", "label"}
                if not required.issubset(entry.keys()):
                    # It might be that some keys are missing or named differently?
                    # Let's print the keys for debugging if it fails
                    raise ValueError(f"Missing required keys in line {line_number}. Found: {list(entry.keys())}")
                data.append(entry)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}")
    print(f"Loaded {len(data)} examples from {file_path}")
    return data


if __name__ == "__main__":
    # Simple sanity‑check when run directly
    import argparse

    parser = argparse.ArgumentParser(description="Load Track A dataset")
    parser.add_argument("path", type=str, help="Path to dev_track_a.jsonl")
    args = parser.parse_args()
    examples = load_track_a(args.path)
    print(f"First example:\n{examples[0]}")
