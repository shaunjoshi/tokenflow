# utils/auth.py
from fastapi import Header, HTTPException
from redis import Redis
import requests
import os
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

redis = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
)


def get_supabase_key_record(api_key: str):
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }

    res = requests.get(
        f"{SUPABASE_URL}/rest/v1/api_keys?key=eq.{api_key}&select=rate_limit,active",
        headers=headers,
    )

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail="Supabase API key lookup failed")

    data = res.json()
    if not data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return data[0]


def get_date_key(api_key: str) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return f"rate:{api_key}:{today}"


def verify_api_key(x_api_key: str = Header(...)) -> str:
    record = get_supabase_key_record(x_api_key)

    if not record["active"]:
        raise HTTPException(status_code=403, detail="API key is disabled")

    limit = record["rate_limit"]
    redis_key = get_date_key(x_api_key)

    usage = redis.get(redis_key)
    usage_count = int(usage or 0)

    if usage_count >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Increment counter
    pipe = redis.pipeline()
    pipe.incr(redis_key)
    pipe.expire(redis_key, 86400)
    pipe.execute()

    return x_api_key
