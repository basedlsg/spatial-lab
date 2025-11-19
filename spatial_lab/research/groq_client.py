"""
Groq API Client for fast LLM inference.

Groq provides extremely fast inference for Llama models.
Used for natural language robot coordination experiments.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class GroqConfig:
    """Configuration for Groq API client"""
    api_key: str
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.1-70b-versatile"  # Fast and capable
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 30


class GroqClient:
    """
    Client for Groq API with fast Llama inference.

    Optimized for:
    - Quick robot coordination decisions
    - Natural language message interpretation
    - Code generation for robot actions
    """

    def __init__(self, config: GroqConfig):
        self.config = config
        self.session = None
        self.request_count = 0
        self.total_latency = 0.0

        logger.info(f"Initialized Groq client with model: {config.model}")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """
        Generate completion from Groq.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Response with content and metadata
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature
        }

        start_time = time.time()
        result = await self._make_request("/chat/completions", payload)
        latency = time.time() - start_time

        self.request_count += 1
        self.total_latency += latency

        # Extract content from response
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            return {
                "content": content,
                "latency_ms": latency * 1000,
                "model": result.get("model", self.config.model),
                "usage": result.get("usage", {})
            }

        raise Exception(f"Invalid response format: {result}")

    async def generate_code(
        self,
        task_description: str,
        available_apis: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate Python code for a robot task.

        Args:
            task_description: Natural language description of what to do
            available_apis: List of available Spatial Lab APIs
            context: Current robot/environment state

        Returns:
            Generated code and metadata
        """
        system_prompt = """You are a robot code generator for Spatial Lab.

Generate Python code that uses only the provided APIs.
The code should be:
- Safe (no file I/O, no network calls, no imports except math)
- Async-compatible (use await for async functions)
- Clear and readable

Return ONLY the Python code, no explanations.
Wrap code in ```python and ``` markers."""

        prompt = f"""Generate code for this robot task:

Task: {task_description}

Available APIs:
{chr(10).join(f'- {api}' for api in available_apis)}

Current Context:
{json.dumps(context, indent=2)}

Generate the Python code:"""

        response = await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for code generation
        )

        # Extract code from response
        content = response["content"]
        code = self._extract_code(content)

        return {
            "code": code,
            "raw_response": content,
            "latency_ms": response["latency_ms"]
        }

    async def interpret_message(
        self,
        message: str,
        sender_context: Dict[str, Any],
        receiver_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Interpret a natural language message from another robot.

        Args:
            message: Natural language message
            sender_context: Information about sending robot
            receiver_context: Information about receiving robot

        Returns:
            Interpreted intent and suggested actions
        """
        system_prompt = """You interpret messages between warehouse robots.

Extract:
1. The sender's need/request
2. What action the receiver should take
3. Priority level (high/medium/low)
4. Any spatial constraints

Return JSON with these fields:
- need: string describing what the sender needs
- suggested_action: string describing what receiver should do
- priority: "high", "medium", or "low"
- spatial_info: any coordinates or locations mentioned
- can_help: boolean if receiver can reasonably help"""

        prompt = f"""Interpret this robot message:

Message: "{message}"

Sender Info:
{json.dumps(sender_context, indent=2)}

Receiver Info:
{json.dumps(receiver_context, indent=2)}

Return JSON interpretation:"""

        response = await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2
        )

        # Parse JSON from response
        try:
            interpretation = json.loads(self._extract_json(response["content"]))
        except json.JSONDecodeError:
            interpretation = {
                "need": "unclear",
                "suggested_action": "wait",
                "priority": "low",
                "spatial_info": None,
                "can_help": False,
                "parse_error": True
            }

        interpretation["latency_ms"] = response["latency_ms"]
        return interpretation

    async def _make_request(self, endpoint: str, payload: Dict) -> Dict[str, Any]:
        """Make async HTTP request to Groq API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.config.base_url}{endpoint}"

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Groq API error {response.status}: {error_text}")
                    raise Exception(f"Groq API error {response.status}: {error_text}")

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling Groq API")
            raise

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks"""
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        return text.strip()

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text"""
        # Try to find JSON in code blocks first
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try to find raw JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]

        return text

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        return {
            "request_count": self.request_count,
            "total_latency_s": self.total_latency,
            "avg_latency_ms": avg_latency * 1000
        }


def create_groq_client(api_key: str, model: str = None) -> GroqConfig:
    """Create Groq configuration"""
    config = GroqConfig(api_key=api_key)
    if model:
        config.model = model
    return config
