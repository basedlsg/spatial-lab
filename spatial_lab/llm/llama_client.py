"""
Llama API Client for Spatial Reasoning

Integrates Meta's Llama API for advanced spatial reasoning and robot coordination tasks.
Supports multimodal input (text + images) and structured JSON outputs.
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
class LlamaAPIConfig:
    """Configuration for Llama API client"""
    api_key: str
    base_url: str = "https://api.llama.com/v1"
    model: str = "Llama-4-Maverick-17B-128E-Instruct-FP8"  # Multimodal model
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 30


class LlamaAPIClient:
    """
    Client for Meta's Llama API with spatial reasoning capabilities.
    
    Supports:
    - Text-based spatial reasoning
    - Multimodal input (text + images)
    - Structured JSON outputs
    - Tool calling for external functions
    - Streaming responses
    """
    
    def __init__(self, config: LlamaAPIConfig):
        self.config = config
        self.session = None
        
        # Validate API key format
        if not config.api_key.startswith("LLM|"):
            raise ValueError("Invalid Llama API key format. Should start with 'LLM|'")
        
        logger.info(f"Initialized Llama API client with model: {config.model}")
    
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
        images: List[str] = None,
        structured_output: Dict = None,
        tools: List[Dict] = None,
        model: str = None
    ) -> Dict[str, Any]:
        """
        Generate spatial reasoning completion with optional multimodal input.
        
        Args:
            prompt: Main user prompt
            system_prompt: System instructions for spatial reasoning
            images: List of image URLs or base64 encoded images
            structured_output: JSON schema for structured response
            tools: List of tool definitions for function calling
            model: Override default model
        
        Returns:
            API response with completion
        """
        
        # Build messages
        messages = []
        
        # Add system prompt for spatial reasoning
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
        
        # Build user message with multimodal content
        user_content = []
        
        # Add text content
        user_content.append({
            "type": "text",
            "text": prompt
        })
        
        # Add images if provided
        if images:
            for image in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": image}
                })
        
        messages.append({
            "role": "user",
            "content": user_content if len(user_content) > 1 else prompt
        })
        
        # Build request payload
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "max_completion_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        # Add structured output if specified
        if structured_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": structured_output
            }
        
        # Add tools if specified
        if tools:
            payload["tools"] = tools
        
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
        
        # Define structured output schema for robot decisions
        decision_schema = {
            "name": "RobotDecision",
            "schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": available_actions,
                        "description": "The action the robot should take"
                    },
                                "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target coordinates [x, y] for move_to action"
                    },
                    "item_id": {
                        "type": "string",
                        "description": "Item ID for pick_item action"
                    },
                    "zone_id": {
                        "type": "string", 
                        "description": "Zone ID for drop_item action"
                    },
                    "message": {
                        "type": "string",
                        "description": "Message content for communicate action"
                    },
                    "target_robot": {
                        "type": "string",
                        "description": "Target robot ID for communicate action"
                    }
                },
                "additionalProperties": true,
                "description": "Parameters for the chosen action"
            },
                    "reasoning": {
                        "type": "string",
                        "description": "Spatial reasoning behind the decision"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in the decision (0-1)"
                    },
                    "coordination_intent": {
                        "type": "string",
                        "description": "How this action coordinates with other robots"
                    }
                },
                "required": ["action", "parameters", "reasoning", "confidence"]
            }
        }
        
        return await self.spatial_reasoning_completion(
            prompt=prompt,
            system_prompt=self._get_robot_coordination_system_prompt(),
            structured_output=decision_schema
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
        """
        
        analysis_schema = {
            "name": "WarehouseAnalysis",
            "schema": {
                "type": "object",
                "properties": {
                    "spatial_efficiency_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "bottlenecks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "severity": {"type": "string"},
                                "recommendation": {"type": "string"}
                            }
                        }
                    },
                    "optimal_zones": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "zone_type": {"type": "string"},
                                "coordinates": {"type": "array"},
                                "purpose": {"type": "string"}
                            }
                        }
                    },
                    "coordination_recommendations": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["spatial_efficiency_score", "bottlenecks", "optimal_zones"]
            }
        }
        
        return await self.spatial_reasoning_completion(
            prompt=prompt,
            structured_output=analysis_schema
        )
    
    async def _make_request(self, endpoint: str, payload: Dict) -> Dict[str, Any]:
        """Make async HTTP request to Llama API"""
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Successful API call to {endpoint}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    raise Exception(f"Llama API error {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling Llama API endpoint {endpoint}")
            raise
        except Exception as e:
            logger.error(f"Error calling Llama API: {e}")
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

Respond with precise, actionable decisions based on spatial analysis."""
    
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


class LlamaAPIManager:
    """
    Manager for multiple Llama API clients with load balancing and error handling.
    """
    
    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Default configuration
        self.config = LlamaAPIConfig(
            api_key=api_key,
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",  # Multimodal model
            max_tokens=2048,
            temperature=0.7
        )
        
        logger.info(f"Initialized Llama API Manager (max_concurrent: {max_concurrent})")
    
    async def get_client(self) -> LlamaAPIClient:
        """Get a configured Llama API client"""
        return LlamaAPIClient(self.config)
    
    async def batch_robot_decisions(
        self,
        robot_requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple robot decision requests concurrently.
        
        Args:
            robot_requests: List of robot decision request data
        
        Returns:
            List of robot decisions
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
                    "completion_message": {
                        "content": {
                            "text": json.dumps({
                                "action": "wait",
                                "parameters": {},
                                "reasoning": f"API error: {str(result)}",
                                "confidence": 0.1
                            })
                        }
                    }
                })
            else:
                processed_results.append(result)
        
        return processed_results


def create_llama_client(api_key: str) -> LlamaAPIManager:
    """
    Factory function to create a Llama API manager.
    
    Args:
        api_key: Llama API key
    
    Returns:
        Configured LlamaAPIManager instance
    """
    return LlamaAPIManager(api_key)


# Test function for API connectivity
async def test_llama_api(api_key: str) -> bool:
    """
    Test Llama API connectivity and basic functionality.
    
    Args:
        api_key: Llama API key to test
    
    Returns:
        True if API is working, False otherwise
    """
    try:
        config = LlamaAPIConfig(api_key=api_key)
        
        async with LlamaAPIClient(config) as client:
            response = await client.spatial_reasoning_completion(
                prompt="Test spatial reasoning: A robot at position (0, 0) needs to reach (5, 5). What's the optimal path?",
                system_prompt="You are testing spatial reasoning capabilities."
            )
            
            if "completion_message" in response:
                logger.info("✅ Llama API test successful")
                return True
            else:
                logger.error("❌ Llama API test failed: Invalid response format")
                return False
                
    except Exception as e:
        logger.error(f"❌ Llama API test failed: {e}")
        return False 