import json
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "pmarb/0.1"}


def get_json(url: str, params: dict | None = None, retries: int = 3, timeout: int = 20):
    if params:
        url = f"{url}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
