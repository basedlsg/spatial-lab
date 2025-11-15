"""
LLM Integration Package for Spatial Reasoning

Provides unified access to multiple LLM providers (Llama, Gemini) with intelligent
routing, load balancing, and spatial reasoning capabilities.
"""

from .llama_client import (
    LlamaAPIClient,
    LlamaAPIManager,
    LlamaAPIConfig,
    create_llama_client,
    test_llama_api
)

from .gemini_client import (
    GeminiAPIClient,
    GeminiAPIManager,
    GeminiAPIConfig,
    create_gemini_client,
    test_gemini_api
)

from .llm_coordinator import (
    LLMCoordinator,
    LLMProvider,
    LLMCapabilities,
    TaskRequirements,
    create_llm_coordinator
)

__all__ = [
    # Llama API
    "LlamaAPIClient",
    "LlamaAPIManager", 
    "LlamaAPIConfig",
    "create_llama_client",
    "test_llama_api",
    
    # Gemini API
    "GeminiAPIClient",
    "GeminiAPIManager",
    "GeminiAPIConfig", 
    "create_gemini_client",
    "test_gemini_api",
    
    # LLM Coordinator
    "LLMCoordinator",
    "LLMProvider",
    "LLMCapabilities",
    "TaskRequirements",
    "create_llm_coordinator"
] 