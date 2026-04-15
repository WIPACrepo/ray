"""resources/test_auth_server.py — run with: uvicorn resources.test_auth_server:app --port 8000"""

from fastapi import Depends, FastAPI
from i3_ray_server.server import Auth

app = FastAPI(title="Auth Test Harness")


@app.get("/test/health")
async def health():
    """Unauthed — confirms the server is up."""
    return {"status": "ok"}


@app.get("/test/auth")
async def authed(_claims: dict = Depends(Auth.require_auth)):
    """Authed — confirms token + roles check passes."""
    return {"status": "ok", "claims": _claims}
