from fastapi import HTTPException
from openai import AsyncOpenAI as AsyncOpenRouter, AsyncOpenAI as AsyncGroqAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import AsyncClient
import logging

log = logging.getLogger(__name__)

app_state = {}

def get_supabase_client() -> AsyncClient:
    client = app_state.get("supabase_client")
    if client is None:
        log.error("Supabase client requested but is not available (initialization failed?).")
        raise HTTPException(status_code=503, detail="Database service temporarily unavailable.")
    return client

def get_bart_classifier():
    classifier = app_state.get("bart_classifier")
    if classifier is None:
        log.error("BART classifier requested but is not available (initialization failed?).")
        raise HTTPException(status_code=503, detail="Text classification service temporarily unavailable.")
    return classifier

def get_openrouter_client() -> AsyncOpenRouter:
    client = app_state.get("openrouter_client")
    if client is None:
        log.error("OpenRouter client requested but is not available (initialization failed?).")
        raise HTTPException(status_code=503, detail="OpenRouter service temporarily unavailable.")
    return client

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    splitter = app_state.get("text_splitter")
    if splitter is None:
        log.error("Text splitter requested but is not available (initialization failed?).")
        raise HTTPException(status_code=503, detail="Text processing service temporarily unavailable.")
    return splitter

def get_lingua_compressor():
    compressor = app_state.get("lingua_compressor")
    if compressor is None:
        log.error("LLMLingua compressor requested but is not available (initialization failed or dependency missing?).")
        raise HTTPException(status_code=503, detail="Compression service temporarily unavailable.")
    return compressor

def get_groq_client() -> AsyncGroqAI:
    client = app_state.get("groq_client")
    if client is None:
        log.error("Groq client requested but is not available (initialization failed?).")
        raise HTTPException(status_code=503, detail="Groq service temporarily unavailable.")
    return client 