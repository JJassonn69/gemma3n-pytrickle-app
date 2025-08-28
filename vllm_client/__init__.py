"""
Local OpenAI-compatible vLLM client wrapper.

Usage:
    from vllm_client import get_client
    client = get_client()
"""

from .client import VLLMClient, get_client

__all__ = ["VLLMClient", "get_client"]
