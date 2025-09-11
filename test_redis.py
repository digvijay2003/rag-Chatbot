#!/usr/bin/env python3
"""
Test basic Redis operations and check current state
"""
import os
import asyncio
import time
from dotenv import load_dotenv
from redis.asyncio import Redis

load_dotenv()

async def test_redis_operations():
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    print(f"Connecting to: {REDIS_URL}")
    
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    
    try:
        # Check current state
        print("\n=== Current Redis State ===")
        all_keys = await redis.keys("*")
        print(f"Existing keys: {len(all_keys)}")
        if all_keys:
            for key in all_keys[:10]:  # Show first 10 keys
                value = await redis.get(key)
                ttl = await redis.ttl(key)
                print(f"  {key} = {value} (TTL: {ttl}s)")
        else:
            print("  No keys found - Redis is empty")
        
        print("\n=== Testing Basic Operations ===")
        
        # Test 1: Simple SET/GET
        await redis.set("test:simple", "hello_world", ex=300)
        value = await redis.get("test:simple")
        print(f"1. SET/GET test: {value}")
        
        # Test 2: Counter operations (used by rate limiter)
        counter_key = "test:counter"
        count1 = await redis.incr(counter_key)
        count2 = await redis.incr(counter_key)
        count3 = await redis.incrby(counter_key, 5)
        print(f"2. Counter tests: {count1}, {count2}, {count3}")
        
        # Test 3: Expiration
        await redis.expire(counter_key, 60)
        ttl = await redis.ttl(counter_key)
        print(f"3. Expiration test: TTL = {ttl}s")
        
        # Test 4: Hash operations
        hash_key = "test:hash"
        await redis.hset(hash_key, "field1", "value1")
        await redis.hset(hash_key, "field2", "value2")
        hash_value = await redis.hgetall(hash_key)
        print(f"4. Hash test: {hash_value}")
        
        # Test 5: List operations
        list_key = "test:list"
        await redis.lpush(list_key, "item1", "item2", "item3")
        list_items = await redis.lrange(list_key, 0, -1)
        print(f"5. List test: {list_items}")
        
        print("\n=== Simulating Rate Limiter Keys ===")
        
        # Create keys similar to what the rate limiter would create
        current_minute = int(time.time() // 60)
        today = time.strftime("%Y-%m-%d")
        
        # Session-based keys
        session_id = "test-session-123"
        await redis.incr(f"rl:session:{session_id}:m:{current_minute}")
        await redis.expire(f"rl:session:{session_id}:m:{current_minute}", 70)
        
        await redis.incr(f"rl:session:{session_id}:d:{today}")
        await redis.expire(f"rl:session:{session_id}:d:{today}", 86400)
        
        # IP-based keys
        ip = "192.168.1.100"
        await redis.incr(f"rl:ip:{ip}:m:{current_minute}")
        await redis.expire(f"rl:ip:{ip}:m:{current_minute}", 70)
        
        # Global keys
        await redis.incr(f"rl:global:embed:m:{current_minute}")
        await redis.expire(f"rl:global:embed:m:{current_minute}", 70)
        
        await redis.incrby(f"rl:global:embed_tokens:m:{current_minute}", 150)
        await redis.expire(f"rl:global:embed_tokens:m:{current_minute}", 70)
        
        print("6. Created rate limiter simulation keys")
        
        # Check what we created
        print("\n=== Checking Rate Limiter Keys ===")
        rl_keys = await redis.keys("rl:*")
        print(f"Rate limiter keys created: {len(rl_keys)}")
        for key in rl_keys:
            value = await redis.get(key)
            ttl = await redis.ttl(key)
            print(f"  {key} = {value} (TTL: {ttl}s)")
        
        print("\n=== Final State ===")
        all_keys_after = await redis.keys("*")
        print(f"Total keys now: {len(all_keys_after)}")
        
        test_keys = await redis.keys("test:*")
        if test_keys:
            await redis.delete(*test_keys)
            print(f"Cleaned up {len(test_keys)} test keys")
        
        print("\n✅ Redis operations test completed successfully!")
        
    except Exception as e:
        print(f"❌ Redis operations failed: {e}")
        return False
    finally:
        await redis.close()
    
    return True

async def check_redis_cli_access():
    """Test if we can access Redis via CLI"""
    print("\n=== Redis CLI Access Test ===")

if __name__ == "__main__":
    success = asyncio.run(test_redis_operations())
    if success:
        asyncio.run(check_redis_cli_access())