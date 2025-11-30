from typing import List, Dict, Any

# The tokenizer can be any object with a `encode` method (e.g., HuggingFace tokenizer).
# We keep the implementation generic so that the same code works for different models.


def tokenize(text: str, tokenizer, max_length: int = 128) -> List[int]:
    """Tokenize a single string.

    Args:
        text: Input text.
        tokenizer: Tokenizer instance with `encode` method.
        max_length: Maximum number of tokens (truncates longer sequences).
    Returns:
        List of token ids.
    """
    # `add_special_tokens` ensures start/end tokens for models that need them.
    token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    # Pad to max_length with tokenizer.pad_token_id if available, otherwise 0.
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    if len(token_ids) < max_length:
        token_ids = token_ids + [pad_id] * (max_length - len(token_ids))
    return token_ids


def prepare_inputs(example: Dict[str, Any], tokenizer, max_length: int = 128) -> Dict[str, List[int]]:
    """Prepare tokenized inputs for a single example.

    Returns a dict with keys ``anchor``, ``choice_a`` and ``choice_b`` containing
    token id lists.
    """
    return {
        "anchor": tokenize(example["anchor"], tokenizer, max_length),
        "choice_a": tokenize(example["choice_a"], tokenizer, max_length),
        "choice_b": tokenize(example["choice_b"], tokenizer, max_length),
        "label": example["label"],
    }


def get_dataset(file_path: str, tokenizer, max_length: int = 128) -> List[Dict[str, Any]]:
    """Load the Track A data and return a list of tokenized examples.

    This function combines :func:`load_track_a` from ``data_loader`` with the
    tokenization utilities defined above.
    """
    from .data_loader import load_track_a

    raw_examples = load_track_a(file_path)
    tokenized = [prepare_inputs(ex, tokenizer, max_length) for ex in raw_examples]
    print(f"Tokenized {len(tokenized)} examples from {file_path}")
    return tokenized
