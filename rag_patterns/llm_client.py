"""
Thin LLM client that supports Ollama (primary) and Together AI (fallback).
All patterns import from here — keeps provider logic in one place.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_prompt(prompt_file: str) -> str:
    path = Path("config/prompts") / prompt_file
    return path.read_text().strip()


class LLMClient:
    def __init__(self, config: dict):
        self.config = config
        llm = config.get("llm", {})
        self.provider = llm.get("provider", "ollama")
        self.model = llm.get("model", "llama3.1:8b-instruct-q8_0")
        self.temperature = llm.get("temperature", 0.0)
        self.max_tokens = llm.get("max_tokens", 512)
        self.seed = llm.get("seed", 42)
        self.base_url = llm.get("base_url", "http://localhost:11434")
        self._system_prompt = _load_prompt("system.txt")

    def complete(self, prompt: str, system: str | None = None) -> tuple[str, int]:
        """
        Call the LLM with the given prompt.
        Returns (answer_text, total_tokens_used).
        """
        sys_prompt = system or self._system_prompt
        if self.provider == "ollama":
            return self._ollama(sys_prompt, prompt)
        elif self.provider in ("together", "groq"):
            return self._together(sys_prompt, prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _ollama(self, system: str, prompt: str) -> tuple[str, int]:
        import ollama as _ollama

        resp = _ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "seed": self.seed,
            },
        )
        text = resp["message"]["content"].strip()
        usage = resp.get("usage", {})
        tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        # Ollama doesn't always return usage; estimate from text length
        if tokens == 0:
            tokens = len(prompt.split()) + len(text.split())
        return text, tokens

    def _together(self, system: str, prompt: str) -> tuple[str, int]:
        import requests

        together_cfg = self.config["llm"].get("together", {})
        api_key = together_cfg.get("api_key") or os.environ.get("TOGETHER_API_KEY", "")
        model = together_cfg.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

        resp = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        return text, tokens
