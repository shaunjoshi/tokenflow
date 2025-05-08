# backend/shared.py
import logging
import os
import sys
from pathlib import Path
import httpx # Keep if needed by other parts or client options

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt # Keep ONLY if used by get_current_user below
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

# --- Import Supabase ASYNC client components (Type Hinting) ---
from supabase import AsyncClient # Only import types needed here

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__) # Logger for this module

# --- Determine Project Root ---
# Assumes shared.py is in backend/, so ../ goes up one level to the root
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # Handle case where __file__ might not be defined (e.g., interactive session)
    BASE_DIR = Path(".").resolve()


# --- Settings Model ---
# Define all required environment variables here
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    VERSION: str = "0.1.0"
    APP_NAME: str = "model-selection-backend"
    
    # JWT
    SUPABASE_JWT_SECRET: SecretStr = Field(..., env="SUPABASE_JWT_SECRET")
    JWT_ALGORITHM: str = "HS256"
    
    # Database (Supabase)
    SUPABASE_URL: str = Field(..., env="SUPABASE_URL") 
    SUPABASE_SERVICE_ROLE_KEY: SecretStr = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")
    
    # OpenRouter
    OPENROUTER_API_KEY: SecretStr = Field(..., env="OPENROUTER_API_KEY")
    OPENROUTER_API_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Groq
    GROQ_API_KEY: SecretStr = Field(..., env="GROQ_API_KEY") 
    GROQ_API_BASE_URL: str = "https://api.groq.com/openai/v1"
    
    # Authentication
    JWT_EXPIRATION_TIME_MINUTES: int = 60 * 24 * 5  # 5 days

    # Llama/Lambda Config
    LAMBDA_API_KEY: SecretStr = Field(..., validation_alias='LAMBDA_API_KEY')
    LAMBDA_API_BASE_URL: str = Field(..., validation_alias='LAMBDA_API_BASE_URL')
    LLAMA_MODEL_NAME: str = Field(default="llama3.1-70b-instruct-fp8", validation_alias='LLAMA_MODEL_NAME') # Example default


    # Tell Pydantic to load from .env file IN THE PROJECT ROOT
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, '.env'), # Construct path to root .env
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra env vars not defined in the model
    )

# --- Instantiate Settings (Single Source of Truth) ---
try:
    log.info("Loading configuration settings...")
    settings = Settings()
    log.info("Configuration loaded successfully.")
    # Log loaded non-secret settings for verification during startup
    log.info(f"Using Supabase URL: {settings.SUPABASE_URL}")
    
    # Log base URLs for API services (but not secrets)
    lambda_base_url = settings.LAMBDA_API_BASE_URL
    log.info(f"Using Lambda Base URL: {lambda_base_url}")
    
    openrouter_base_url = settings.OPENROUTER_API_BASE_URL
    log.info(f"Using OpenRouter Base URL: {openrouter_base_url}")
    
    # Log model name if available
    if hasattr(settings, 'LLAMA_MODEL_NAME'):
        log.info(f"Using Llama Model: {settings.LLAMA_MODEL_NAME}")
except Exception as e:
    log.critical(f"CRITICAL: Failed to load configuration settings: {e}")
    # Exit if essential settings are missing/invalid on load
    sys.exit(f"Configuration Error: {e}")


# --- DECLARE Shared Clients Placeholder (Initialized in main.py startup) ---
log.info("Declaring shared client variable placeholder (supabase)...")
# This variable *can* be updated by main.py's lifespan, but using app state is safer
supabase: AsyncClient | None = None

# (Declare other placeholders like embedding_function if needed by endpoints IN main.py)
# embedding_function = None
# log.warning("Placeholder for embedding_function initialization - update if needed by /upload.")


# --- Security Utilities ---
security = HTTPBearer()
log.info("Security utilities initialized.")


# --- Shared Dependencies ---
# This dependency uses the settings object loaded above
async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Dependency to validate Supabase JWT (using SECRET from settings)
    and return user info.
    """
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Check if the secret needed for validation was loaded successfully
    if not settings.SUPABASE_JWT_SECRET:
        log.error("SUPABASE_JWT_SECRET is not configured for JWT validation.")
        raise HTTPException(
            status_code=500, detail="Authentication configuration error (Missing Secret)"
        )

    try:
        # Use the secret from the loaded settings object
        jwt_secret = settings.SUPABASE_JWT_SECRET
        if hasattr(jwt_secret, 'get_secret_value'):
            jwt_secret = jwt_secret.get_secret_value()
            
        payload = jwt.decode(
            token,
            jwt_secret, # Access secret value
            algorithms=[settings.JWT_ALGORITHM],
            audience="authenticated" # Standard Supabase audience
        )
        user_id: str | None = payload.get("sub")
        if user_id is None:
            log.warning("JWT validation failed: 'sub' (user ID) claim missing.")
            raise credentials_exception

        log.debug(f"JWT validated for user_id: {user_id}")
        # Return essential info (consider adding roles if present in payload)
        return {"id": user_id, "email": payload.get("email")}
    except JWTError as e:
        log.warning(f"JWT Error during validation: {e}")
        raise credentials_exception
    except Exception as e:
        # Log unexpected errors during decoding/validation
        log.exception(f"Unexpected error during JWT validation: {e}") # Log traceback
        raise credentials_exception


# --- Export necessary items ---
# Export the instantiated settings object and shared dependencies/constants
__all__ = [
    "settings",
    # "supabase", # Avoid exporting the placeholder if using app_state/Depends
    "security",
    "get_current_user",
    # "vader_analyzer", # Remove VADER sentiment analyzer
    # "embedding_function", # Export if initialized and needed elsewhere
    # "CHROMA_BASE_PATH", # Export constants if needed
]

log.info("Shared module loaded.") # Note: Client initialization moved to main.py lifespan