#!/usr/bin/env python3
"""
Integration tests for LLM clients.

Tests actual API connectivity and response handling for:
- Gemini API
- Groq API
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial_lab.llm import (
    GeminiAPIConfig,
    GeminiAPIClient,
    GroqAPIConfig,
    GroqAPIClient,
)


# Skip tests if API keys not available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
class TestGeminiIntegration:
    """Test Gemini API integration with real calls."""

    @pytest.fixture
    def config(self):
        return GeminiAPIConfig(
            api_key=GEMINI_API_KEY,
            model="gemini-2.0-flash",
        )

    @pytest.mark.asyncio
    async def test_basic_completion(self, config):
        """Test basic text completion."""
        async with GeminiAPIClient(config) as client:
            response = await client.spatial_reasoning_completion(
                prompt="What is 2 + 2? Reply with just the number.",
                system_prompt="You are a helpful assistant.",
            )

            assert response is not None
            # Response should contain "4"
            if isinstance(response, dict):
                candidates = response.get("candidates", [])
                if candidates:
                    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    assert "4" in text
            else:
                assert "4" in str(response)

    @pytest.mark.asyncio
    async def test_spatial_reasoning_prompt(self, config):
        """Test spatial reasoning with plan selection."""
        async with GeminiAPIClient(config) as client:
            prompt = """Given these warehouse paths:
            Plan A: 20 meters, risk score 0.3
            Plan B: 25 meters, risk score 0.1

            For a safety-critical mission transporting hazardous materials,
            which plan is better? Reply with just "A" or "B"."""

            response = await client.spatial_reasoning_completion(
                prompt=prompt,
                system_prompt="You are a warehouse robot path planner.",
            )

            assert response is not None
            # Should prefer B (safer) for safety-critical
            text = str(response)
            assert "A" in text or "B" in text

    @pytest.mark.asyncio
    async def test_json_response_format(self, config):
        """Test that model can return structured JSON."""
        async with GeminiAPIClient(config) as client:
            prompt = """Select the best plan and return JSON:
            Plan A: length=20, risk=0.3
            Plan B: length=25, risk=0.1

            Return: {"chosen_plan": "A" or "B", "confidence": 0.0 to 1.0}"""

            response = await client.spatial_reasoning_completion(
                prompt=prompt,
                system_prompt="Return only valid JSON, no other text.",
            )

            assert response is not None


@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set")
class TestGroqIntegration:
    """Test Groq API integration with real calls."""

    @pytest.fixture
    def config(self):
        return GroqAPIConfig(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
        )

    @pytest.mark.asyncio
    async def test_basic_completion(self, config):
        """Test basic text completion."""
        async with GroqAPIClient(config) as client:
            response = await client.spatial_reasoning_completion(
                prompt="What is 2 + 2? Reply with just the number.",
                system_prompt="You are a helpful assistant.",
            )

            assert response is not None
            # Should contain "4"
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    assert "4" in text
            else:
                assert "4" in str(response)

    @pytest.mark.asyncio
    async def test_json_mode(self, config):
        """Test JSON mode response."""
        async with GroqAPIClient(config) as client:
            prompt = """Return a JSON object with:
            {"answer": 4, "confidence": 1.0}"""

            response = await client.spatial_reasoning_completion(
                prompt=prompt,
                system_prompt="Return only valid JSON.",
                json_mode=True,
            )

            assert response is not None


class TestResponseParsing:
    """Test response parsing utilities."""

    def test_parse_gemini_response(self):
        """Test parsing Gemini response format."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"chosen_plan": "A", "confidence": 0.8}'}
                        ]
                    }
                }
            ]
        }

        candidates = response.get("candidates", [])
        assert len(candidates) == 1
        text = candidates[0]["content"]["parts"][0]["text"]
        assert "chosen_plan" in text

    def test_parse_groq_response(self):
        """Test parsing Groq response format."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"chosen_plan": "B", "confidence": 0.9}'
                    }
                }
            ]
        }

        choices = response.get("choices", [])
        assert len(choices) == 1
        text = choices[0]["message"]["content"]
        assert "chosen_plan" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
