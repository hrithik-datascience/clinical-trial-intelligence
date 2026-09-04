"""Shared Anthropic client for every agent.

One place to get this right: ANTHROPIC_WORKSPACE_ID is required for
identity-linked API keys (Section 6, confirmed against the live API — the SDK
does not read it automatically, it must be sent as the anthropic-workspace-id
header on every request). Get the client wrong here once, not once per agent.

DEVIATION FROM PLAN, found running Module 7's live test three times in a row:
one run in five failed with the exact "anthropic-workspace-id is required"
400 error, despite the header being set correctly and identical requests
succeeding immediately before and after. Confirmed not a config bug (the env
var loads consistently across 5 fresh processes) and not a client-construction
bug (5/5 calls succeeded reusing one client in a tight loop). This is rare,
non-deterministic infrastructure flakiness, not a code defect — the correct
response is a narrow retry, not further debugging of something that reproduces
0/10 on demand. Retrying is scoped tightly to this one known message rather
than all 400s, so a genuinely malformed request still fails immediately.
"""

from __future__ import annotations

import os
import time

from anthropic import Anthropic, BadRequestError
from dotenv import load_dotenv

from src import audit

load_dotenv()

_MODEL = os.getenv("CTI_MODEL", "claude-sonnet-5")

EFFORT_ROUTING = os.getenv("CTI_EFFORT_ROUTING", "low")
EFFORT_EXTRACTION = os.getenv("CTI_EFFORT_EXTRACTION", "medium")
EFFORT_REASONING = os.getenv("CTI_EFFORT_REASONING", "high")

_FLAKY_WORKSPACE_ERROR = "anthropic-workspace-id is required"
_MAX_ATTEMPTS = 3


def get_client() -> Anthropic:
    """The one place a raw Anthropic() should be constructed in this project."""
    headers = {}
    if ws_id := os.getenv("ANTHROPIC_WORKSPACE_ID"):
        headers["anthropic-workspace-id"] = ws_id
    return Anthropic(default_headers=headers)


def model_name() -> str:
    return _MODEL


def parse_with_retry(
    client: Anthropic,
    /,
    *,
    agent: str,
    run_id: str | None = None,
    retrieved_chunk_ids: list[str] | None = None,
    **kwargs,
):
    """client.messages.parse(**kwargs), retrying only the known-flaky
    workspace-id 400 (see module docstring). Any other error — including a
    genuinely malformed request — is raised on the first attempt.

    Module 12: every call is also logged as an LLM_CALL audit event —
    latency (including any retry backoff, since that's the real wall time a
    caller experienced) and token counts read straight off the response's
    own `usage` field, never estimated (T-24). `agent` is required, not
    inferred, so a future call site that forgets to tag itself fails loudly
    (a missing keyword argument) rather than logging silently as "unknown".
    `run_id` is optional: a call made outside a Supervisor run (e.g. an
    agent exercised directly in a script or test) is still logged, tagged
    "standalone" rather than skipped.
    """
    started = time.monotonic()
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = client.messages.parse(**kwargs)
        except BadRequestError as exc:
            if _FLAKY_WORKSPACE_ERROR not in str(exc) or attempt == _MAX_ATTEMPTS:
                raise
            time.sleep(0.5 * attempt)
            continue

        messages = kwargs.get("messages") or [{}]
        input_preview = str(messages[-1].get("content", ""))
        audit.log_llm_call(
            run_id=run_id or "standalone",
            agent=agent,
            model=kwargs.get("model", model_name()),
            effort=str((kwargs.get("output_config") or {}).get("effort", "")),
            latency_s=time.monotonic() - started,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            retrieved_chunk_ids=retrieved_chunk_ids,
            input_preview=input_preview,
            output_preview=str(response.parsed_output),
        )
        return response
