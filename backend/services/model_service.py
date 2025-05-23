from backend.models.rankings import MODEL_RANKINGS

async def select_model_for_category(category: str) -> str:
    """
    Select the most appropriate Groq model based on the prompt category.
    Returns the model ID to be used with Groq API.
    """
    # Check if category exists in MODEL_RANKINGS
    if category.lower() in MODEL_RANKINGS:
        # Get the primary model for this category
        for model in MODEL_RANKINGS[category.lower()]["models"]:
            if model["primary"]:
                return model["id"]

        # If no primary model found, use the first one
        if MODEL_RANKINGS[category.lower()]["models"]:
            return MODEL_RANKINGS[category.lower()]["models"][0]["id"]

    # Default fallback if category not found or no models defined
    return "llama-3.1-8b-instant" 