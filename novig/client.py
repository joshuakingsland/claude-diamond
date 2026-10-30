"""A thin signed REST client for the Novig v3 API.

Paper is the default host. Real money needs ``NOVIG_ENV=live`` on purpose;
nothing falls through to production by accident.

    NOVIG_ENV       paper (default) or live
    NOVIG_KEY_ID    the trading key's UUID (its subaccount's id)
    NOVIG_PEM       path to that key's private PEM
"""

import json
import os
import time
import urllib.parse

import requests

from novig.signing import load_private_key, sign, string_to_sign

HOSTS = {
    "paper": "https://api.paper.novig.com",
    "live": "https://api.novig.com",
}


class NovigError(Exception):
    def __init__(self, status, code, message, body=None):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.body = body


def encode_query(params):
    """A query string in canonical order, so what is sent is what is signed.

    List values become repeated names; ``None`` values are dropped.
    """
    pairs = []
    for name, value in (params or {}).items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if item is None:
                continue
            pairs.append((urllib.parse.quote(str(name), safe="-._~"),
                          urllib.parse.quote(str(item), safe="-._~")))
    pairs.sort()
    return "&".join(f"{name}={value}" for name, value in pairs)


class NovigClient:
    def __init__(self, key_id, private_key, env="paper", timeout=5.0,
                 session=None):
        if env not in HOSTS:
            raise ValueError(f"unknown Novig environment {env!r}")
        self.env = env
        self.host = HOSTS[env]
        self.key_id = key_id
        self.private_key = private_key
        self.timeout = timeout
        # One keep-alive session: a fresh TLS handshake per order is the
        # single largest avoidable latency in this loop.
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls, env=None, **kwargs):
        env = env or os.environ.get("NOVIG_ENV", "paper")
        with open(os.environ["NOVIG_PEM"], "rb") as handle:
            private_key = load_private_key(handle.read())
        return cls(os.environ["NOVIG_KEY_ID"], private_key, env=env, **kwargs)

    # ------------------------------------------------------------ transport
    def signed_headers(self, method, path, query="", data=b""):
        ts = str(int(time.time() * 1000))
        return {
            "Novig-Key-Id": self.key_id,
            "Novig-Timestamp": ts,
            "Novig-Signature": sign(
                self.private_key,
                string_to_sign(ts, method, path, query, data)),
        }

    def connect_ws(self, **kwargs):
        """An open, signed websocket on ``/v3/ws`` (needs ``websockets``)."""
        from websockets.sync.client import connect
        url = "wss://" + self.host.split("://", 1)[1] + "/v3/ws"
        return connect(url, additional_headers=self.signed_headers(
            "GET", "/v3/ws"), max_size=None, **kwargs)

    def call(self, method, path, params=None, body=None):
        query = encode_query(params)
        data = (b"" if body is None
                else json.dumps(body, separators=(",", ":")).encode("utf-8"))
        headers = self.signed_headers(method, path, query, data)
        # Without a content type the server hashes zero bytes, so a body
        # would never verify.
        headers["Content-Type"] = "application/json"
        url = self.host + path + (f"?{query}" if query else "")
        response = self.session.request(method.upper(), url, data=data,
                                        headers=headers, timeout=self.timeout)
        try:
            payload = response.json()
        except ValueError:
            # The edge refuses with an HTML 403 that never reaches the API.
            payload = None
        if response.status_code >= 400:
            payload = payload if isinstance(payload, dict) else {}
            raise NovigError(response.status_code,
                             payload.get("code", "HTTP_ERROR"),
                             payload.get("message", response.text[:200]),
                             payload)
        return payload

    # ------------------------------------------------------------- helpers
    def server_time(self):
        return self.call("GET", "/v3/time")["ts"]

    def echo(self, body):
        return self.call("POST", "/v3/echo", body=body)

    def leagues(self):
        return self.call("GET", "/v3/types/leagues")

    def market_types(self):
        return self.call("GET", "/v3/types/markets")

    def _pages(self, path, params):
        params = dict(params)
        while True:
            page = self.call("GET", path, params)
            yield from page.get("items", [])
            if not page.get("next"):
                return
            params["after"] = page["next"]

    def markets(self, **filters):
        """Every market matching ``filters``; comma-join list filters."""
        params = {k: ",".join(v) if isinstance(v, (list, tuple)) else v
                  for k, v in filters.items()}
        params.setdefault("limit", 5000)
        return list(self._pages("/v3/catalog/markets", params))

    def events(self, **filters):
        params = {k: ",".join(v) if isinstance(v, (list, tuple)) else v
                  for k, v in filters.items()}
        return list(self._pages("/v3/catalog/events", params))

    def market(self, market_id):
        return self.call("GET", f"/v3/catalog/markets/{market_id}")

    def book(self, market_id, depth=None):
        return self.call("GET", f"/v3/catalog/markets/{market_id}/book",
                         {"depth": depth})

    def place_order(self, outcome_id, price, qty, tif, client_id=None):
        """Place once. A replayed place always places again: never retry
        this blindly, read ``orders``/``fills`` first."""
        body = {"outcomeId": outcome_id, "price": price, "qty": int(qty),
                "tif": tif}
        if client_id:
            body["clientId"] = client_id
        return self.call("POST", "/v3/orders", body=body)

    def orders(self, **filters):
        return self.call("GET", "/v3/orders", filters)

    def fills(self, **filters):
        return self.call("GET", "/v3/portfolio/fills", filters)

    def positions(self):
        return self.call("GET", "/v3/portfolio/positions")
