"""
Text Simplifier
---------------
Calls LM Studio (local) to convert English → ASL-grammar-friendly text.

ASL grammar rules applied:
- Drop articles (a, an, the)
- Drop copulas (is, are, was)
- Use present tense only
- Topic-comment word order
- Short sentences (max 8 words)
- Simple, common vocabulary
"""

import requests
import json
import config


class TextSimplifier:
    def __init__(self):
        self.url     = config.LM_STUDIO_URL
        self.headers = {"Content-Type": "application/json"}

    def simplify(self, text: str) -> str:
        """
        Convert English text to ASL-friendly simplified gloss order.
        Falls back to basic rule-based cleanup if LM Studio is unreachable.
        """
        try:
            result = self._call_lm_studio(text)
            if result and result.strip():
                return result.strip().upper()
        except Exception as e:
            print(f"LM Studio unavailable ({e}), using rule-based fallback.")

        return self._rule_based_simplify(text)

    def is_available(self) -> bool:
        """Check if LM Studio is reachable."""
        try:
            r = requests.get(
                config.LM_STUDIO_URL.replace("/v1/chat/completions", "/v1/models"),
                timeout=3
            )
            return r.status_code == 200
        except Exception:
            return False

    # ── Private ───────────────────────────────────────────

    def _call_lm_studio(self, text: str) -> str:
        payload = {
            "model": "local-model",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You convert English sentences into ASL (American Sign Language) gloss order. "
                        "Rules:\n"
                        "1. Remove articles: a, an, the\n"
                        "2. Remove copulas: is, are, was, were, am, be\n"
                        "3. Use present tense only (no -ing, -ed unless necessary)\n"
                        "4. Keep max 8 words per sentence\n"
                        "5. Use simple, common words\n"
                        "6. Topic goes first (e.g. 'STORE GO' not 'GO TO STORE')\n"
                        "7. Output ONLY the simplified words in uppercase, nothing else. "
                        "No punctuation, no explanation."
                    )
                },
                {
                    "role": "user",
                    "content": f"Convert to ASL gloss: {text}"
                }
            ],
            "temperature": 0.1,
            "max_tokens": 80,
        }
        resp   = requests.post(
            self.url, headers=self.headers,
            data=json.dumps(payload), timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _rule_based_simplify(self, text: str) -> str:
        """
        Basic rule-based ASL simplification when LM Studio is unavailable.
        Removes common function words that ASL doesn't use.
        """
        drop = {
            "a", "an", "the", "is", "are", "was", "were", "am",
            "be", "been", "being", "will", "would", "could", "should",
            "have", "has", "had", "do", "does", "did",
            "to", "of", "in", "on", "at", "by", "for", "with",
            "and", "but", "or", "so", "because", "that", "which",
            "very", "really", "just", "also",
        }
        words   = text.lower().split()
        cleaned = [w.upper() for w in words
                   if ''.join(c for c in w if c.isalpha()) not in drop]
        return " ".join(cleaned) if cleaned else text.upper()