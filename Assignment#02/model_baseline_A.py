import os
import json
import hashlib
from enum import Enum
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

# Cache file path
CACHE_FILE = Path("openai_cache.json")


class ResponseEnum(str, Enum):
    A = "A"
    B = "B"


class SimilarityPrediction(BaseModel):
    explanation: str
    closer: ResponseEnum


class GPTBaseline:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model_name = model_name
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self):
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_cache_key(self, anchor: str, choice_a: str, choice_b: str) -> str:
        # Create a unique hash for the input triplet
        content = f"{anchor}|{choice_a}|{choice_b}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def predict(self, anchor: str, choice_a: str, choice_b: str) -> ResponseEnum:
        cache_key = self._get_cache_key(anchor, choice_a, choice_b)
        if cache_key in self.cache:
            # Return cached result
            return ResponseEnum(self.cache[cache_key])

        # Few-shot examples
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert on narrative understanding and storytelling. "
                    "Your task is to identify which of two candidate stories (Story A or Story B) "
                    "is more narratively similar to a given Anchor Story. "
                    "Consider themes, plot structure, character archetypes, and narrative tone. "
                    "Provide a brief explanation followed by your choice (A or B)."
                ),
            },
            # Example 1
            {
                "role": "user",
                "content": (
                    "Anchor Story: The book follows an international organization named the Ministry for the Future...\n"
                    "Story A: The old grandmother Tina arrives in town to attend the wedding...\n"
                    "Story B: The nano-plague that poisoned Earth's water supply has reached its 60-year critical mass...\n\n"
                    "Which story is more similar to the Anchor Story?"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "explanation": "The Anchor Story deals with global threats (climate change) and future survival. Story B also deals with a global threat (nano-plague) and future survival in a sci-fi setting, whereas Story A is a domestic comedy/drama. Thus, Story B is more similar.",
                    "closer": "B"
                })
            },
            # Example 2
            {
                "role": "user",
                "content": (
                    "Anchor Story: Glenn Tyler (Elvis Presley), a childish 25-year old, gets into a fight...\n"
                    "Story A: Bill Babbitt supported the death penalty, until it came knocking at his door...\n"
                    "Story B: A white-collar suburban father Kyle is surprised at his office by long-lost college buddy Zack...\n\n"
                    "Which story is more similar to the Anchor Story?"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "explanation": "The Anchor Story is about a troubled young man on probation who is falsely suspected but eventually redeems himself. Story A involves a troubled brother with mental health issues and a difficult family choice, sharing themes of family trouble and legal/moral dilemmas. Story B is about a bizarre psychodrama retreat. Story A shares more thematic elements regarding troubled youth/family dynamics.",
                    "closer": "A"
                })
            },
            # Actual Query
            {
                "role": "user",
                "content": (
                    f"Anchor Story: {anchor}\n\n"
                    f"Story A: {choice_a}\n\n"
                    f"Story B: {choice_b}\n\n"
                    "Which story is more similar to the Anchor Story?"
                ),
            },
        ]

        try:
            completion = self.client.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=SimilarityPrediction,
            )
            prediction = completion.choices[0].message.parsed
            result = prediction.closer

            # Update cache
            self.cache[cache_key] = result
            self._save_cache()
            
            return result
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Fallback or re-raise. For now, let's re-raise to be safe.
            raise e

if __name__ == "__main__":
    # Test run
    model = GPTBaseline()
    anchor = "A hero saves the princess from a dragon."
    choice_a = "A knight fights a monster to rescue a royal."
    choice_b = "A chef cooks a delicious meal for the king."
    result = model.predict(anchor, choice_a, choice_b)
    print(f"Prediction: {result}")
