"""
Groq API Client for Spatial Reasoning

Uses Groq's fast inference API (OpenAI-compatible) for spatial reasoning tasks.
Supports Llama, Mixtral, and Gemma models via Groq's infrastructure.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union
import aiohttp
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GroqAPIConfig:
    """Configuration for Groq API client"""
    api_key: str
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"  # Fast, capable model
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 60


# Available Groq models
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Best for complex reasoning tasks",
    "llama-3.1-70b-versatile": "Previous gen, still excellent",
    "llama-3.1-8b-instant": "Fast, good for simple tasks",
    "mixtral-8x7b-32768": "Good for long context",
    "gemma2-9b-it": "Google's Gemma model",
}


class GroqAPIClient:
    """
    Client for Groq API with spatial reasoning capabilities.

    Groq provides extremely fast inference for open-source models.
    Uses OpenAI-compatible API format.
    """

    def __init__(self, config: GroqAPIConfig):
        self.config = config
        self.session = None

        logger.info(f"Initialized Groq API client with model: {config.model}")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def spatial_reasoning_completion(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Generate spatial reasoning completion with Groq.

        Args:
            prompt: Main user prompt
            system_prompt: System instructions for spatial reasoning
            model: Override default model
            json_mode: Whether to request JSON output

        Returns:
            API response with completion
        """

        # Build messages
        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        else:
            messages.append({
                "role": "system",
                "content": self._get_default_spatial_system_prompt()
            })

        # Add user message
        messages.append({
            "role": "user",
            "content": prompt
        })

        # Build request payload
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }

        # Add JSON mode if requested
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        # Make API call
        return await self._make_request("/chat/completions", payload)

    async def robot_coordination_decision(
        self,
        robot_id: str,
        observation: Dict[str, Any],
        task_description: str,
        available_actions: List[str],
        nearby_robots: List[Dict] = None,
        warehouse_layout: Dict = None
    ) -> Dict[str, Any]:
        """
        Generate robot coordination decision using spatial reasoning.

        Args:
            robot_id: Unique robot identifier
            observation: Current robot observation data
            task_description: Description of the task to accomplish
            available_actions: List of actions the robot can take
            nearby_robots: Information about nearby robots
            warehouse_layout: Warehouse layout information

        Returns:
            Structured decision with action, parameters, and reasoning
        """

        # Create spatial reasoning prompt
        prompt = self._build_robot_decision_prompt(
            robot_id, observation, task_description,
            available_actions, nearby_robots, warehouse_layout
        )

        # Add JSON output instructions
        prompt += """

You MUST respond with ONLY a valid JSON object (no markdown, no explanation before/after):
{
    "action": "one of the available actions",
    "parameters": {
        "destination": [x, y] for move_to,
        "item_id": "string" for pick_item,
        "zone_id": "string" for drop_item
    },
    "reasoning": "your spatial reasoning explanation",
    "confidence": 0.85,
    "coordination_intent": "how this coordinates with other robots"
}
"""

        response = await self.spatial_reasoning_completion(
            prompt=prompt,
            system_prompt=self._get_robot_coordination_system_prompt(),
            json_mode=True
        )

        return response

    async def analyze_warehouse_layout(
        self,
        layout_data: Dict[str, Any],
        optimization_goals: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze warehouse layout for spatial optimization opportunities.
        """

        prompt = f"""
Analyze this warehouse layout for spatial efficiency and robot coordination:

Layout Data: {json.dumps(layout_data, indent=2)}

Optimization Goals: {optimization_goals or ['efficiency', 'safety', 'throughput']}

Provide detailed spatial analysis including:
1. Traffic flow patterns
2. Bottleneck identification
3. Optimal robot positioning
4. Coordination zones
5. Safety considerations
6. Efficiency improvements

Respond with a JSON object containing your analysis.
"""

        return await self.spatial_reasoning_completion(
            prompt=prompt,
            system_prompt="You are a warehouse layout optimization expert. Always respond in JSON format.",
            json_mode=True
        )

    async def _make_request(self, endpoint: str, payload: Dict) -> Dict[str, Any]:
        """Make async HTTP request to Groq API"""

        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.config.base_url}{endpoint}"

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Successful Groq API call to {endpoint}")
                    return result
                elif response.status == 429:
                    # Rate limited - return error for fallback
                    error_text = await response.text()
                    logger.warning(f"Groq rate limited: {error_text}")
                    raise Exception(f"Rate limited: {error_text}")
                else:
                    error_text = await response.text()
                    logger.error(f"Groq API error {response.status}: {error_text}")
                    raise Exception(f"Groq API error {response.status}: {error_text}")

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling Groq API endpoint {endpoint}")
            raise
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise

    def _get_default_spatial_system_prompt(self) -> str:
        """Default system prompt for spatial reasoning tasks"""
        return """You are an advanced spatial reasoning AI specialized in warehouse robot coordination.

Your capabilities include:
- 3D spatial understanding and navigation
- Multi-robot coordination and collision avoidance
- Efficient path planning and task optimization
- Real-time decision making under uncertainty
- Safety-first approach to all operations

Always provide clear reasoning for spatial decisions and consider:
- Robot positions and movements
- Obstacle avoidance and safety
- Task efficiency and coordination
- Resource optimization
- Communication with other robots

Respond with precise, actionable decisions based on spatial analysis.
Always format your responses as valid JSON when requested."""

    def _get_robot_coordination_system_prompt(self) -> str:
        """System prompt specifically for robot coordination decisions"""
        return """You are a robot coordination specialist in a virtual warehouse environment.

Your role is to make optimal decisions for individual robots while considering:
- Current robot position and capabilities
- Nearby robots and their intentions
- Task requirements and priorities
- Warehouse layout and constraints
- Safety and efficiency optimization

Make decisions that:
1. Accomplish the assigned task efficiently
2. Avoid collisions with other robots
3. Optimize overall warehouse throughput
4. Maintain safety protocols
5. Coordinate effectively with the robot fleet

Always respond with valid JSON containing your decision."""

    def _build_robot_decision_prompt(
        self,
        robot_id: str,
        observation: Dict[str, Any],
        task_description: str,
        available_actions: List[str],
        nearby_robots: List[Dict] = None,
        warehouse_layout: Dict = None
    ) -> str:
        """Build detailed prompt for robot decision making"""

        prompt = f"""
ROBOT COORDINATION DECISION REQUEST

Robot ID: {robot_id}
Task: {task_description}

Current Observation:
{json.dumps(observation, indent=2)}

Available Actions: {', '.join(available_actions)}
"""

        if nearby_robots:
            prompt += f"\n\nNearby Robots:\n{json.dumps(nearby_robots, indent=2)}"

        if warehouse_layout:
            # Truncate layout if too long
            layout_str = json.dumps(warehouse_layout, indent=2)
            if len(layout_str) > 2000:
                layout_str = layout_str[:2000] + "\n... (truncated)"
            prompt += f"\n\nWarehouse Layout Summary:\n{layout_str}"

        prompt += """

Based on this information, make an optimal decision that:
1. Progresses toward the task goal efficiently
2. Avoids conflicts with other robots
3. Maintains safety protocols
4. Optimizes coordination with the fleet
"""

        return prompt


class GroqAPIManager:
    """
    Manager for Groq API clients with load balancing and error handling.
    """

    def __init__(self, api_key: str, model: str = None, max_concurrent: int = 5):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Default configuration
        self.config = GroqAPIConfig(
            api_key=api_key,
            model=model or "llama-3.3-70b-versatile",
            max_tokens=2048,
            temperature=0.7
        )

        logger.info(f"Initialized Groq API Manager with model: {self.config.model}")

    async def get_client(self) -> GroqAPIClient:
        """Get a configured Groq API client"""
        return GroqAPIClient(self.config)

    async def batch_robot_decisions(
        self,
        robot_requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple robot decision requests concurrently.
        """

        async def process_request(request_data):
            async with self.semaphore:
                async with await self.get_client() as client:
                    return await client.robot_coordination_decision(**request_data)

        # Process all requests concurrently
        tasks = [process_request(request) for request in robot_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing robot request {i}: {result}")
                # Return fallback decision
                processed_results.append({
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "action": "wait",
                                "parameters": {},
                                "reasoning": f"API error: {str(result)}",
                                "confidence": 0.1
                            })
                        }
                    }]
                })
            else:
                processed_results.append(result)

        return processed_results


def create_groq_client(api_key: str, model: str = None) -> GroqAPIManager:
    """
    Factory function to create a Groq API manager.

    Args:
        api_key: Groq API key
        model: Model to use (default: llama-3.3-70b-versatile)

    Returns:
        Configured GroqAPIManager instance
    """
    return GroqAPIManager(api_key, model)


# Test function for API connectivity
async def test_groq_api(api_key: str, model: str = None) -> bool:
    """
    Test Groq API connectivity and basic functionality.

    Args:
        api_key: Groq API key to test
        model: Model to test with

    Returns:
        True if API is working, False otherwise
    """
    try:
        config = GroqAPIConfig(
            api_key=api_key,
            model=model or "llama-3.3-70b-versatile"
        )

        async with GroqAPIClient(config) as client:
            response = await client.spatial_reasoning_completion(
                prompt="A robot at position (0, 0) needs to reach (5, 5). What direction should it move? Respond in one sentence.",
                system_prompt="You are testing spatial reasoning. Be brief."
            )

            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                logger.info(f"Groq API test successful. Response: {content[:100]}...")
                return True
            else:
                logger.error(f"Groq API test failed: Invalid response format: {response}")
                return False

    except Exception as e:
        logger.error(f"Groq API test failed: {e}")
        return False
