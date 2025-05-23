DEFAULT_MODEL_CATEGORIES = [
    "reasoning",
    "function-calling",
    "text-to-text",
    "multilingual",
    "nsfw",
]

MODEL_RANKINGS = {
    "reasoning": {
        "title": "Reasoning",
        "description": "Logic, problem-solving, and analysis",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "primary": True},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "primary": False},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "primary": False},
        ],
    },
    "function-calling": {
        "title": "Function Calling",
        "description": "API interactions and tool usage",
        "models": [
            {"id": "qwen-qwq-32b", "name": "Quen QWQ", "primary": True},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "primary": False},
        ],
    },
    "text-to-text": {
        "title": "Text to Text",
        "description": "General text generation and transformation",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "primary": True},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "primary": False},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "primary": False},
        ],
    },
    "multilingual": {
        "title": "Multilingual",
        "description": "Cross-language generation and translation",
        "models": [
            {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "name": "Llama 4", "primary": True},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "primary": False},
        ],
    },
    "nsfw": {
        "title": "NSFW Detection",
        "description": "Content moderation and safety evaluation",
        "models": [
            {"id": "llama-guard-3-8b", "name": "Llama Guard 3 8B", "primary": True},
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "primary": False},
        ],
    },
} 