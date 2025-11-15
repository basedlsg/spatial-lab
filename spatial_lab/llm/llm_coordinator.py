"""
LLM Coordinator for Spatial Reasoning

Manages multiple LLM APIs (Llama, Gemini) with intelligent routing, load balancing,
and fallback capabilities for optimal spatial reasoning performance.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from .llama_client import LlamaAPIManager, create_llama_client, test_llama_api
from .gemini_client import GeminiAPIManager, create_gemini_client, test_gemini_api

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers"""
    LLAMA = "llama"
    GEMINI = "gemini"


@dataclass
class LLMCapabilities:
    """Capabilities of different LLM providers"""
    multimodal: bool = False
    structured_output: bool = False
    tool_calling: bool = False
    max_tokens: int = 2048
    cost_per_token: float = 0.0001  # Approximate cost
    avg_latency_ms: int = 1000


@dataclass
class TaskRequirements:
    """Requirements for a specific spatial reasoning task"""
    requires_multimodal: bool = False
    requires_structured_output: bool = False
    requires_tool_calling: bool = False
    priority: str = "normal"  # low, normal, high, critical
    max_latency_ms: int = 5000
    cost_sensitivity: str = "normal"  # low, normal, high


class LLMCoordinator:
    """
    Coordinates multiple LLM providers for optimal spatial reasoning performance.
    
    Features:
    - Intelligent routing based on task requirements
    - Load balancing across providers
    - Automatic fallback on failures
    - Performance monitoring and optimization
    - Cost optimization
    """
    
    def __init__(
        self,
        llama_api_key: str = None,
        gemini_api_key: str = None,
        preferred_provider: LLMProvider = LLMProvider.LLAMA
    ):
        self.providers = {}
        self.capabilities = {}
        self.performance_stats = {}
        self.preferred_provider = preferred_provider
        
        # Initialize available providers
        if llama_api_key:
            self.providers[LLMProvider.LLAMA] = create_llama_client(llama_api_key)
            self.capabilities[LLMProvider.LLAMA] = LLMCapabilities(
                multimodal=True,
                structured_output=True,
                tool_calling=True,
                max_tokens=4096,
                cost_per_token=0.0002,
                avg_latency_ms=1200
            )
            self.performance_stats[LLMProvider.LLAMA] = {
                "total_requests": 0,
                "successful_requests": 0,
                "avg_latency_ms": 0,
                "error_count": 0
            }
        
        if gemini_api_key:
            self.providers[LLMProvider.GEMINI] = create_gemini_client(gemini_api_key)
            self.capabilities[LLMProvider.GEMINI] = LLMCapabilities(
                multimodal=True,
                structured_output=False,
                tool_calling=True,
                max_tokens=2048,
                cost_per_token=0.00015,
                avg_latency_ms=800
            )
            self.performance_stats[LLMProvider.GEMINI] = {
                "total_requests": 0,
                "successful_requests": 0,
                "avg_latency_ms": 0,
                "error_count": 0
            }
        
        if not self.providers:
            raise ValueError("At least one LLM provider API key must be provided")
        
        logger.info(f"Initialized LLM Coordinator with providers: {list(self.providers.keys())}")
    
    async def test_all_providers(self) -> Dict[LLMProvider, bool]:
        """Test connectivity to all configured providers"""
        results = {}
        
        for provider in self.providers:
            if provider == LLMProvider.LLAMA:
                results[provider] = await test_llama_api(
                    self.providers[provider].api_key
                )
            elif provider == LLMProvider.GEMINI:
                results[provider] = await test_gemini_api(
                    self.providers[provider].api_key
                )
        
        logger.info(f"Provider test results: {results}")
        return results
    
    def select_provider(
        self,
        task_requirements: TaskRequirements,
        exclude_providers: List[LLMProvider] = None
    ) -> LLMProvider:
        """
        Select the optimal provider based on task requirements and current performance.
        
        Args:
            task_requirements: Requirements for the task
            exclude_providers: Providers to exclude from selection
        
        Returns:
            Selected LLM provider
        """
        exclude_providers = exclude_providers or []
        available_providers = [p for p in self.providers.keys() if p not in exclude_providers]
        
        if not available_providers:
            raise RuntimeError("No available providers for task")
        
        # Score each provider based on requirements
        provider_scores = {}
        
        for provider in available_providers:
            capabilities = self.capabilities[provider]
            stats = self.performance_stats[provider]
            
            score = 100  # Base score
            
            # Check hard requirements
            if task_requirements.requires_multimodal and not capabilities.multimodal:
                score = 0
            if task_requirements.requires_structured_output and not capabilities.structured_output:
                score = 0
            if task_requirements.requires_tool_calling and not capabilities.tool_calling:
                score = 0
            
            if score == 0:
                continue
            
            # Performance scoring
            if stats["total_requests"] > 0:
                success_rate = stats["successful_requests"] / stats["total_requests"]
                score *= success_rate
                
                # Latency scoring
                if task_requirements.max_latency_ms > 0:
                    latency_score = max(0, 1 - (stats["avg_latency_ms"] / task_requirements.max_latency_ms))
                    score *= (0.7 + 0.3 * latency_score)
            
            # Cost scoring
            if task_requirements.cost_sensitivity == "high":
                cost_score = 1 / (capabilities.cost_per_token * 1000 + 1)
                score *= (0.8 + 0.2 * cost_score)
            
            # Preferred provider bonus
            if provider == self.preferred_provider:
                score *= 1.1
            
            provider_scores[provider] = score
        
        if not provider_scores:
            # Fallback to first available provider
            return available_providers[0]
        
        # Select provider with highest score
        selected_provider = max(provider_scores, key=provider_scores.get)
        
        logger.debug(f"Selected provider {selected_provider} with score {provider_scores[selected_provider]:.2f}")
        return selected_provider
    
    async def robot_coordination_decision(
        self,
        robot_id: str,
        observation: Dict[str, Any],
        task_description: str,
        available_actions: List[str],
        nearby_robots: List[Dict] = None,
        warehouse_layout: Dict = None,
        task_requirements: TaskRequirements = None
    ) -> Tuple[Dict[str, Any], LLMProvider]:
        """
        Generate robot coordination decision using the optimal provider.
        
        Returns:
            Tuple of (decision_response, provider_used)
        """
        
        task_requirements = task_requirements or TaskRequirements(
            requires_structured_output=True,
            priority="normal"
        )
        
        # Try providers with fallback
        attempted_providers = []
        
        while len(attempted_providers) < len(self.providers):
            try:
                provider = self.select_provider(task_requirements, attempted_providers)
                attempted_providers.append(provider)
                
                start_time = time.time()
                
                # Make the API call
                if provider == LLMProvider.LLAMA:
                    async with await self.providers[provider].get_client() as client:
                        response = await client.robot_coordination_decision(
                            robot_id=robot_id,
                            observation=observation,
                            task_description=task_description,
                            available_actions=available_actions,
                            nearby_robots=nearby_robots,
                            warehouse_layout=warehouse_layout
                        )
                elif provider == LLMProvider.GEMINI:
                    async with await self.providers[provider].get_client() as client:
                        response = await client.robot_coordination_decision(
                            robot_id=robot_id,
                            observation=observation,
                            task_description=task_description,
                            available_actions=available_actions,
                            nearby_robots=nearby_robots,
                            warehouse_layout=warehouse_layout
                        )
                
                # Update performance stats
                latency_ms = int((time.time() - start_time) * 1000)
                self._update_performance_stats(provider, latency_ms, success=True)
                
                logger.debug(f"Robot decision completed by {provider} in {latency_ms}ms")
                return response, provider
                
            except Exception as e:
                logger.warning(f"Provider {provider} failed for robot decision: {e}")
                self._update_performance_stats(provider, 0, success=False)
                
                if len(attempted_providers) >= len(self.providers):
                    # All providers failed
                    logger.error("All providers failed for robot coordination decision")
                    # Return fallback decision
                    fallback_response = {
                        "completion_message": {
                            "content": {
                                "text": json.dumps({
                                    "action": "wait",
                                    "parameters": {},
                                    "reasoning": "All LLM providers failed, using fallback",
                                    "confidence": 0.1,
                                    "coordination_intent": "Waiting for system recovery"
                                })
                            }
                        }
                    }
                    return fallback_response, None
        
        raise RuntimeError("Failed to get robot coordination decision from any provider")
    
    async def analyze_warehouse_layout(
        self,
        layout_data: Dict[str, Any],
        optimization_goals: List[str] = None,
        task_requirements: TaskRequirements = None
    ) -> Tuple[Dict[str, Any], LLMProvider]:
        """
        Analyze warehouse layout using the optimal provider.
        
        Returns:
            Tuple of (analysis_response, provider_used)
        """
        
        task_requirements = task_requirements or TaskRequirements(
            priority="normal",
            cost_sensitivity="normal"
        )
        
        # Try providers with fallback
        attempted_providers = []
        
        while len(attempted_providers) < len(self.providers):
            try:
                provider = self.select_provider(task_requirements, attempted_providers)
                attempted_providers.append(provider)
                
                start_time = time.time()
                
                # Make the API call
                if provider == LLMProvider.LLAMA:
                    async with await self.providers[provider].get_client() as client:
                        response = await client.analyze_warehouse_layout(
                            layout_data=layout_data,
                            optimization_goals=optimization_goals
                        )
                elif provider == LLMProvider.GEMINI:
                    async with await self.providers[provider].get_client() as client:
                        response = await client.analyze_warehouse_layout(
                            layout_data=layout_data,
                            optimization_goals=optimization_goals
                        )
                
                # Update performance stats
                latency_ms = int((time.time() - start_time) * 1000)
                self._update_performance_stats(provider, latency_ms, success=True)
                
                logger.debug(f"Warehouse analysis completed by {provider} in {latency_ms}ms")
                return response, provider
                
            except Exception as e:
                logger.warning(f"Provider {provider} failed for warehouse analysis: {e}")
                self._update_performance_stats(provider, 0, success=False)
        
        raise RuntimeError("Failed to analyze warehouse layout with any provider")
    
    async def batch_robot_decisions(
        self,
        robot_requests: List[Dict[str, Any]],
        task_requirements: TaskRequirements = None
    ) -> List[Tuple[Dict[str, Any], LLMProvider]]:
        """
        Process multiple robot decisions concurrently across providers.
        
        Args:
            robot_requests: List of robot decision request data
            task_requirements: Task requirements for all requests
        
        Returns:
            List of (response, provider_used) tuples
        """
        
        task_requirements = task_requirements or TaskRequirements(
            requires_structured_output=True,
            priority="normal"
        )
        
        # Distribute requests across available providers
        provider_list = list(self.providers.keys())
        distributed_requests = []
        
        for i, request in enumerate(robot_requests):
            provider = provider_list[i % len(provider_list)]
            distributed_requests.append((request, provider))
        
        # Process requests concurrently
        async def process_request(request_data, provider):
            try:
                response, actual_provider = await self.robot_coordination_decision(
                    **request_data,
                    task_requirements=task_requirements
                )
                return response, actual_provider
            except Exception as e:
                logger.error(f"Batch request failed: {e}")
                # Return fallback
                fallback_response = {
                    "completion_message": {
                        "content": {
                            "text": json.dumps({
                                "action": "wait",
                                "parameters": {},
                                "reasoning": f"Batch processing error: {str(e)}",
                                "confidence": 0.1
                            })
                        }
                    }
                }
                return fallback_response, None
        
        tasks = [
            process_request(request, provider) 
            for request, provider in distributed_requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing exception: {result}")
                fallback_response = {
                    "completion_message": {
                        "content": {
                            "text": json.dumps({
                                "action": "wait",
                                "parameters": {},
                                "reasoning": f"Exception: {str(result)}",
                                "confidence": 0.1
                            })
                        }
                    }
                }
                processed_results.append((fallback_response, None))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _update_performance_stats(
        self,
        provider: LLMProvider,
        latency_ms: int,
        success: bool
    ):
        """Update performance statistics for a provider"""
        
        stats = self.performance_stats[provider]
        stats["total_requests"] += 1
        
        if success:
            stats["successful_requests"] += 1
            # Update rolling average latency
            if stats["avg_latency_ms"] == 0:
                stats["avg_latency_ms"] = latency_ms
            else:
                stats["avg_latency_ms"] = int(
                    0.8 * stats["avg_latency_ms"] + 0.2 * latency_ms
                )
        else:
            stats["error_count"] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all providers"""
        
        report = {
            "providers": {},
            "summary": {
                "total_requests": 0,
                "total_successful": 0,
                "total_errors": 0,
                "avg_success_rate": 0
            }
        }
        
        total_requests = 0
        total_successful = 0
        total_errors = 0
        
        for provider, stats in self.performance_stats.items():
            success_rate = 0
            if stats["total_requests"] > 0:
                success_rate = stats["successful_requests"] / stats["total_requests"]
            
            report["providers"][provider.value] = {
                "total_requests": stats["total_requests"],
                "successful_requests": stats["successful_requests"],
                "error_count": stats["error_count"],
                "success_rate": success_rate,
                "avg_latency_ms": stats["avg_latency_ms"],
                "capabilities": {
                    "multimodal": self.capabilities[provider].multimodal,
                    "structured_output": self.capabilities[provider].structured_output,
                    "tool_calling": self.capabilities[provider].tool_calling,
                    "max_tokens": self.capabilities[provider].max_tokens
                }
            }
            
            total_requests += stats["total_requests"]
            total_successful += stats["successful_requests"]
            total_errors += stats["error_count"]
        
        report["summary"]["total_requests"] = total_requests
        report["summary"]["total_successful"] = total_successful
        report["summary"]["total_errors"] = total_errors
        
        if total_requests > 0:
            report["summary"]["avg_success_rate"] = total_successful / total_requests
        
        return report


def create_llm_coordinator(
    llama_api_key: str = None,
    gemini_api_key: str = None,
    preferred_provider: str = "llama"
) -> LLMCoordinator:
    """
    Factory function to create an LLM coordinator.
    
    Args:
        llama_api_key: Llama API key
        gemini_api_key: Gemini API key
        preferred_provider: Preferred provider ("llama" or "gemini")
    
    Returns:
        Configured LLMCoordinator instance
    """
    
    preferred = LLMProvider.LLAMA if preferred_provider.lower() == "llama" else LLMProvider.GEMINI
    
    return LLMCoordinator(
        llama_api_key=llama_api_key,
        gemini_api_key=gemini_api_key,
        preferred_provider=preferred
    ) 