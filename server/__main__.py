"""Production entry point: one process (rooms live in memory), hardened uvicorn settings."""
import os

import uvicorn

if __name__ == "__main__":
    behind_proxy = os.getenv("SANTORINI_TRUST_PROXY", "") == "1"
    uvicorn.run("server.app:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), workers=1,
                # Hosts that terminate TLS at a proxy need its X-Forwarded-* headers; trust them only when told to.
                proxy_headers=behind_proxy, forwarded_allow_ips="*" if behind_proxy else "127.0.0.1",
                ws_max_size=4096, server_header=False, timeout_keep_alive=15)
