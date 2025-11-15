"""
Gemini API Client for Spatial Reasoning

Integrates Google's Gemini API as a secondary LLM for spatial reasoning tasks.
Provides fallback capabilities and alternative reasoning approaches.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union
import aiohttp
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeminiAPIConfig:
    """Configuration for Gemini API client"""
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    model: str = "gemini-1.5-pro"
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 30


class GeminiAPIClient:
    """
    Client for Google's Gemini API with spatial reasoning capabilities.
    
    Supports:
    - Text-based spatial reasoning
    - Multimodal input (text + images)
    - Structured outputs
    - Function calling
    """
    
    def __init__(self, config: GeminiAPIConfig):
        self.config = config
        self.session = None
        
        logger.info(f"Initialized Gemini API client with model: {config.model}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={
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
        images: List[str] = None,
        model: str = None
    ) -> Dict[str, Any]:
        """
        Generate spatial reasoning completion with Gemini.
        
        Args:
            prompt: Main user prompt
            system_prompt: System instructions for spatial reasoning
            images: List of image URLs or base64 encoded images
            model: Override default model
        
        Returns:
            API response with completion
        """
        
        # Build the content
        parts = []
        
        # Add system prompt if provided
        if system_prompt:
            parts.append({"text": f"System: {system_prompt}\n\nUser: {prompt}"})
        else:
            parts.append({"text": prompt})
        
        # Add images if provided (Gemini format)
        if images:
            for image in images:
                if image.startswith("data:"):
                    # Base64 encoded image
                    mime_type, image_data = image.split(",", 1)
                    mime_type = mime_type.split(":")[1].split(";")[0]
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_data
                        }
                    })
                else:
                    # URL - would need to fetch and convert to base64
                    parts.append({"text": f"[Image URL: {image}]"})
        
        # Build request payload
        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "maxOutputTokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
        }
        
        # Make API call
        model_name = model or self.config.model
        endpoint = f"/models/{model_name}:generateContent"
        
        return await self._make_request(endpoint, payload)
    
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
        Generate robot coordination decision using Gemini.
        
        Args:
            robot_id: Unique robot identifier
            observation: Current robot observation data
            task_description: Description of the task to accomplish
            available_actions: List of actions the robot can take
            nearby_robots: Information about nearby robots
            warehouse_layout: Warehouse layout information
        
        Returns:
            Decision with action, parameters, and reasoning
        """
        
        # Create spatial reasoning prompt
        prompt = self._build_robot_decision_prompt(
            robot_id, observation, task_description, 
            available_actions, nearby_robots, warehouse_layout
        )
        
        prompt += """
        
        Respond with a JSON object containing:
        {
            "action": "chosen_action_from_available_list",
            "parameters": {"key": "value"},
            "reasoning": "detailed spatial reasoning",
            "confidence": 0.85,
            "coordination_intent": "how this coordinates with other robots"
        }
        """
        
        return await self.spatial_reasoning_completion(
            prompt=prompt,
            system_prompt=self._get_robot_coordination_system_prompt()
        )
    
    async def analyze_warehouse_layout(
        self,
        layout_data: Dict[str, Any],
        optimization_goals: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze warehouse layout for spatial optimization opportunities.
        
        Args:
            layout_data: Warehouse layout information
            optimization_goals: Specific optimization objectives
        
        Returns:
            Analysis with recommendations and spatial insights
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
            system_prompt="You are a warehouse layout optimization expert."
        )
    
    async def _make_request(self, endpoint: str, payload: Dict) -> Dict[str, Any]:
        """Make async HTTP request to Gemini API"""
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.config.base_url}{endpoint}?key={self.config.api_key}"
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Successful API call to {endpoint}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    raise Exception(f"Gemini API error {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling Gemini API endpoint {endpoint}")
            raise
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise
    
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

Provide structured decisions with clear reasoning and high confidence scores."""
    
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
            prompt += f"\n\nWarehouse Layout:\n{json.dumps(warehouse_layout, indent=2)}"
        
        prompt += """
        
        Based on this information, make an optimal decision that:
        1. Progresses toward the task goal efficiently
        2. Avoids conflicts with other robots
        3. Maintains safety protocols
        4. Optimizes coordination with the fleet
        
        Provide your decision with detailed spatial reasoning.
        """
        
        return prompt


class GeminiAPIManager:
    """
    Manager for Gemini API clients with error handling.
    """
    
    def __init__(self, api_key: str, max_concurrent: int = 3):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Default configuration
        self.config = GeminiAPIConfig(
            api_key=api_key,
            model="gemini-1.5-pro",
            max_tokens=2048,
            temperature=0.7
        )
        
        logger.info(f"Initialized Gemini API Manager (max_concurrent: {max_concurrent})")
    
    async def get_client(self) -> GeminiAPIClient:
        """Get a configured Gemini API client"""
        return GeminiAPIClient(self.config)


def create_gemini_client(api_key: str) -> GeminiAPIManager:
    """
    Factory function to create a Gemini API manager.
    
    Args:
        api_key: Gemini API key
    
    Returns:
        Configured GeminiAPIManager instance
    """
    return GeminiAPIManager(api_key)


# Test function for API connectivity
async def test_gemini_api(api_key: str) -> bool:
    """
    Test Gemini API connectivity and basic functionality.
    
    Args:
        api_key: Gemini API key to test
    
    Returns:
        True if API is working, False otherwise
    """
    try:
        config = GeminiAPIConfig(api_key=api_key)
        
        async with GeminiAPIClient(config) as client:
            response = await client.spatial_reasoning_completion(
                prompt="Test spatial reasoning: A robot at position (0, 0) needs to reach (5, 5). What's the optimal path?",
                system_prompt="You are testing spatial reasoning capabilities."
            )
            
            if "candidates" in response and len(response["candidates"]) > 0:
                logger.info("✅ Gemini API test successful")
                return True
            else:
                logger.error("❌ Gemini API test failed: Invalid response format")
                return False
                
    except Exception as e:
        logger.error(f"❌ Gemini API test failed: {e}")
        return False 