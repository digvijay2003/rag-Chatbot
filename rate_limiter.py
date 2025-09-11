import os
import time
import math
from typing import Optional
from fastapi import HTTPException, status, Request
from redis.asyncio import Redis
from basic_logging import rate_limit_logger

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = Redis.from_url(REDIS_URL, decode_responses=True)

GOOGLE_EMBED_RPM = int(os.getenv("GOOGLE_EMBED_RPM", "100"))   
GOOGLE_EMBED_RPD = int(os.getenv("GOOGLE_EMBED_RPD", "1000"))  
GOOGLE_EMBED_TPM = int(os.getenv("GOOGLE_EMBED_TPM", "30000")) 

SESSION_RPM = int(os.getenv("SESSION_RPM", "1"))   
SESSION_RPD = int(os.getenv("SESSION_RPD", "500"))  
SESSION_TPM = int(os.getenv("SESSION_TPM", "3000")) 

IP_RPM = int(os.getenv("IP_RPM", "1"))
IP_RPD = int(os.getenv("IP_RPD", "200"))
IP_TPM = int(os.getenv("IP_TPM", "3000"))

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))

async def get_client_ip(request: Request) -> str:
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        ip = xff.split(",")[0].strip()
        rate_limit_logger.info(f"IP from X-Forwarded-For: {ip}")
        return ip
    
    client = request.client
    ip = client.host if client else "unknown"
    rate_limit_logger.info(f"IP from client: {ip}")
    return ip

async def rate_limit_dependency(request: Request, session_id: Optional[str], text_for_token_estimate: Optional[str] = None):
    """
    Enhanced rate limiter with debugging
    """
    now = time.time()
    minute_slot = int(now // 60)
    today_str = time.strftime("%Y-%m-%d")

    ip_identity = await get_client_ip(request)
    identity = session_id if session_id else ip_identity
    
    rate_limit_logger.info(f"Rate limiting - Session ID: {session_id}, IP: {ip_identity}, Using identity: {identity}")
    
    if session_id:
        rpm_limit = SESSION_RPM
        rpd_limit = SESSION_RPD
        tpm_limit = SESSION_TPM
        limit_type = "session"
        rate_limit_logger.info(f"Using SESSION limits: RPM={rpm_limit}, RPD={rpd_limit}, TPM={tpm_limit}")
    else:
        rpm_limit = IP_RPM
        rpd_limit = IP_RPD
        tpm_limit = IP_TPM
        limit_type = "ip"
        rate_limit_logger.info(f"Using IP limits: RPM={rpm_limit}, RPD={rpd_limit}, TPM={tpm_limit}")

    min_key = f"rl:{limit_type}:{identity}:m:{minute_slot}"
    day_key = f"rl:{limit_type}:{identity}:d:{today_str}"
    
    rate_limit_logger.info(f"Redis keys - Minute: {min_key}, Day: {day_key}")

    global_embed_min_key = f"rl:global:embed:m:{minute_slot}"
    global_embed_day_key = f"rl:global:embed:d:{today_str}"
    global_embed_token_min_key = f"rl:global:embed_tokens:m:{minute_slot}"

    current_min = await redis.get(min_key)
    current_day = await redis.get(day_key)
    rate_limit_logger.info(f"Current counts before increment - Minute: {current_min}, Day: {current_day}")

    cur_min = await redis.incr(min_key)
    if cur_min == 1:
        await redis.expire(min_key, 70)
    
    rate_limit_logger.info(f"After increment - Minute count: {cur_min}/{rpm_limit}")
    
    if cur_min > rpm_limit:
        rate_limit_logger.warning(f"RPM limit exceeded: {cur_min}/{rpm_limit} for identity {identity}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {rpm_limit} requests per minute (identity={identity}, type={limit_type})"
        )

    cur_day = await redis.incr(day_key)
    if cur_day == 1:
        await redis.expire(day_key, 60*60*48)
        
    rate_limit_logger.info(f"After increment - Daily count: {cur_day}/{rpd_limit}")
    
    if cur_day > rpd_limit:
        rate_limit_logger.warning(f"Daily limit exceeded: {cur_day}/{rpd_limit} for identity {identity}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily limit exceeded: {rpd_limit} requests per day (identity={identity}, type={limit_type})"
        )

    cur_global_embed_min = await redis.incr(global_embed_min_key)
    if cur_global_embed_min == 1:
        await redis.expire(global_embed_min_key, 70)
    if cur_global_embed_min > GOOGLE_EMBED_RPM:
        await redis.decr(min_key)
        rate_limit_logger.warning(f"Global embedding RPM limit exceeded: {cur_global_embed_min}/{GOOGLE_EMBED_RPM}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Service busy: global embedding RPM limit reached. Try again later."
        )

    cur_global_embed_day = await redis.incr(global_embed_day_key)
    if cur_global_embed_day == 1:
        await redis.expire(global_embed_day_key, 60*60*48)
    if cur_global_embed_day > GOOGLE_EMBED_RPD:
        await redis.decr(global_embed_min_key)
        await redis.decr(day_key)
        await redis.decr(min_key)
        rate_limit_logger.warning(f"Global embedding daily limit exceeded: {cur_global_embed_day}/{GOOGLE_EMBED_RPD}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Service busy: global embedding daily limit reached. Try again later."
        )

    if text_for_token_estimate:
        tokens = estimate_tokens(text_for_token_estimate)
        rate_limit_logger.info(f"Estimated tokens: {tokens}")
        
        cur_token_min = await redis.incrby(global_embed_token_min_key, tokens)
        if cur_token_min == tokens:
            await redis.expire(global_embed_token_min_key, 70)
        if cur_token_min > GOOGLE_EMBED_TPM:
            await redis.decrby(global_embed_token_min_key, tokens)
            await redis.decr(global_embed_min_key)
            await redis.decr(global_embed_day_key)
            await redis.decr(day_key)
            await redis.decr(min_key)
            rate_limit_logger.warning(f"Global token limit exceeded: {cur_token_min}/{GOOGLE_EMBED_TPM}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Service busy: global token-per-minute limit exceeded (try again later)."
            )

        session_token_min_key = f"rl:{limit_type}:{identity}:tokens:m:{minute_slot}"
        cur_sess_tokens = await redis.incrby(session_token_min_key, tokens)
        if cur_sess_tokens == tokens:
            await redis.expire(session_token_min_key, 70)
            
        rate_limit_logger.info(f"Token count for identity: {cur_sess_tokens}/{tpm_limit}")
        
        if cur_sess_tokens > tpm_limit:
            await redis.decrby(session_token_min_key, tokens)
            await redis.decrby(global_embed_token_min_key, tokens)
            await redis.decr(global_embed_min_key)
            await redis.decr(global_embed_day_key)
            await redis.decr(day_key) 
            await redis.decr(min_key)
            rate_limit_logger.warning(f"Session token limit exceeded: {cur_sess_tokens}/{tpm_limit}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"{limit_type.title()} token-per-minute limit exceeded ({tpm_limit} tokens)"
            )

    rate_limit_logger.info(f"Rate limit check passed for identity: {identity} (type: {limit_type})")
    return True