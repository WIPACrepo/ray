"""resources/test_auth_client.py — hit the auth test harness with a real Keycloak token.

Requires the test harness to be running:
    uvicorn resources.test_auth_server:app --port 8000
"""

import asyncio
import json
import logging
from pathlib import Path

from rest_tools.client import SavedDeviceGrantAuth

LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Test auth endpoints against the test harness."""
    server_url = "http://localhost:8000"
    client_id = "ray-serve"
    token_url = "https://keycloak.icecube.wisc.edu/auth/realms/IceCube"

    rc = SavedDeviceGrantAuth(
        server_url,
        token_url=token_url,
        filename=str(Path(f"~/device-refresh-token-{client_id}-test").expanduser()),
        client_id=client_id,
        retries=0,
    )

    results: list[dict[str, str]] = []

    # 1. Health (no auth needed)
    LOGGER.info("GET /test/health (no auth)...")
    resp = await rc.request("GET", "/test/health")
    LOGGER.info(f"  -> {json.dumps(resp, indent=2)}")
    results.append({"endpoint": "/test/health", "auth": "none", "status": "✅ pass"})

    # 2. Auth check (requires valid token + role)
    LOGGER.info("GET /test/auth (with token)...")
    resp = await rc.request("GET", "/test/auth")
    LOGGER.info(f"  -> {json.dumps(resp, indent=2)}")
    claims = resp.get("claims", {})
    results.append({"endpoint": "/test/auth", "auth": "bearer", "status": "✅ pass"})

    # Summary
    roles = claims.get("resource_access", {}).get(client_id, {}).get("roles", [])
    print("\n=== Auth Test Summary ===")
    print(f"  Server:    {server_url}")
    print(f"  Client ID: {client_id}")
    print(f"  Subject:   {claims.get('sub', '?')}")
    print(f"  Issuer:    {claims.get('iss', '?')}")
    print(f"  Roles:     {', '.join(roles) or '(none)'}")
    print()
    print(f"  {'Endpoint':<20} {'Auth':<10} {'Result'}")
    print(f"  {'─' * 20} {'─' * 10} {'─' * 10}")
    for r in results:
        print(f"  {r['endpoint']:<20} {r['auth']:<10} {r['status']}")
    print()

    LOGGER.info("All checks passed!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)8s] %(name)s — %(message)s",
    )
    asyncio.run(main())
