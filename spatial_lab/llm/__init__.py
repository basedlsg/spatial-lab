"""
LLM Integration Package for Spatial Reasoning

Provides unified access to multiple LLM providers (Groq, Gemini, Llama) with intelligent
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

from .groq_client import (
    GroqAPIClient,
    GroqAPIManager,
    GroqAPIConfig,
    create_groq_client,
    test_groq_api,
    GROQ_MODELS
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

    # Groq API
    "GroqAPIClient",
    "GroqAPIManager",
    "GroqAPIConfig",
    "create_groq_client",
    "test_groq_api",
    "GROQ_MODELS",

    # LLM Coordinator
    "LLMCoordinator",
    "LLMProvider",
    "LLMCapabilities",
    "TaskRequirements",
    "create_llm_coordinator"
] 