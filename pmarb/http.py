import json
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "pmarb/0.1"}
STATS = {"calls": 0, "retries": 0, "errors": 0}


def get_json(url: str, params: dict | None = None, retries: int = 3, timeout: int = 20):
    """GET JSON con reintentos. Cada reintento y cada error final se registra en stderr
    (líneas 'HTTP-RETRY' / 'HTTP-ERROR') para que aparezcan en el log aunque no salgan en el resumen."""
    if params:
        url = f"{url}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"
    STATS["calls"] += 1
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == retries - 1:
                STATS["errors"] += 1
                print(f"HTTP-ERROR {url} -> {e!r}", file=sys.stderr)
                raise
            STATS["retries"] += 1
            print(f"HTTP-RETRY {i + 1}/{retries - 1} {url} -> {e!r}", file=sys.stderr)
            time.sleep(2 ** i)
