from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute=60, requests_per_hour=1000):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests = defaultdict(list)
        self.hour_requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        
        # Clean old requests
        now = datetime.now()
        self.minute_requests[client_ip] = [
            req_time for req_time in self.minute_requests[client_ip]
            if req_time > now - timedelta(minutes=1)
        ]
        self.hour_requests[client_ip] = [
            req_time for req_time in self.hour_requests[client_ip]
            if req_time > now - timedelta(hours=1)
        ]
        
        # Check limits
        if len(self.minute_requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(status_code=429, detail="Too many requests per minute")
        
        if len(self.hour_requests[client_ip]) >= self.requests_per_hour:
            raise HTTPException(status_code=429, detail="Too many requests per hour")
        
        # Log request
        self.minute_requests[client_ip].append(now)
        self.hour_requests[client_ip].append(now)
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        response.headers["X-Process-Time"] = str(time.time() - start_time)
        
        return response