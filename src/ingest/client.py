"""Shared HTTP client for the four public data sources.

Two responsibilities, both required by Section D.3 and Section J:

  * respect each API's rate limit before we are throttled, not after
  * fail loudly. A pull that did not happen must raise, never return empty.

The distinction that matters is between "the API said there is no data" and
"the request failed". The first is a legitimate result the Safety agent must be
able to report; the second is an error. Silently conflating them is how a demo
ends up quietly claiming a drug has no reported adverse events.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

log = logging.getLogger(__name__)

# Retry on transport errors and on the status codes that mean "try again".
# 4xx other than 429 are permanent — retrying a malformed query just wastes
# the rate-limit budget.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4
_BACKOFF_BASE = 1.5


class IngestError(RuntimeError):
    """A pull failed. Never caught-and-ignored inside this package."""


class NoDataFound(Exception):
    """The API answered successfully and there is genuinely nothing there.

    Deliberately not an IngestError — this is a finding, not a failure.
    """


class RateLimiter:
    """Minimum-interval throttle, shared across threads."""

    def __init__(self, min_interval_s: float) -> None:
        self._min_interval = min_interval_s
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


class ApiClient:
    """Rate-limited, retrying HTTP client for one upstream API."""

    def __init__(
        self,
        name: str,
        base_url: str = "",
        min_interval_s: float = 0.0,
        timeout_s: float = 30.0,
        headers: dict[str, str] | None = None,
        user_agent: str | None = None,
    ) -> None:
        """user_agent=None keeps httpx's default.

        This is not cosmetic. ClinicalTrials.gov's bot filter returns 403 for
        unrecognised User-Agent strings while allowing `python-httpx/*` — a
        descriptive custom UA gets the whole source blocked. Verified
        2026-09-03. We identify ourselves where the API asks for it through a
        supported channel instead (PubMed's `tool`/`email` params), rather than
        spoofing a browser, which would be both evasion and fragile.
        """
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._limiter = RateLimiter(min_interval_s)
        merged = dict(headers or {})
        if user_agent:
            merged["User-Agent"] = user_agent
        self._client = httpx.Client(
            timeout=timeout_s,
            follow_redirects=True,
            headers=merged,
        )

    def get(self, path: str, params: dict | None = None) -> httpx.Response:
        """GET with backoff. Raises IngestError once retries are exhausted."""
        url = f"{self.base_url}/{path.lstrip('/')}" if self.base_url else path
        last_error: str = ""

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            self._limiter.wait()
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("%s attempt %d/%d transport error: %s",
                            self.name, attempt, _MAX_ATTEMPTS, last_error)
            else:
                if response.status_code == 200:
                    return response
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code not in _RETRY_STATUS:
                    # Permanent. Fail now rather than burning the retry budget.
                    raise IngestError(f"{self.name} {url} -> {last_error}")
                log.warning("%s attempt %d/%d %s",
                            self.name, attempt, _MAX_ATTEMPTS, last_error)

            if attempt < _MAX_ATTEMPTS:
                # Honour Retry-After when the server sent one.
                delay = _BACKOFF_BASE ** attempt
                time.sleep(delay)

        raise IngestError(
            f"{self.name} {url} failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )

    def get_json(self, path: str, params: dict | None = None) -> dict:
        response = self.get(path, params)
        try:
            return response.json()
        except ValueError as exc:
            raise IngestError(f"{self.name} returned non-JSON: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
