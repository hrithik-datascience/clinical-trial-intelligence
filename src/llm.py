"""Shared Anthropic client for every agent.

One place to get this right: ANTHROPIC_WORKSPACE_ID is required for
identity-linked API keys (Section 6, confirmed against the live API — the SDK
does not read it automatically, it must be sent as the anthropic-workspace-id
header on every request). Get the client wrong here once, not once per agent.
"""

from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_MODEL = os.getenv("CTI_MODEL", "claude-sonnet-5")

EFFORT_ROUTING = os.getenv("CTI_EFFORT_ROUTING", "low")
EFFORT_EXTRACTION = os.getenv("CTI_EFFORT_EXTRACTION", "medium")
EFFORT_REASONING = os.getenv("CTI_EFFORT_REASONING", "high")


def get_client() -> Anthropic:
    """The one place a raw Anthropic() should be constructed in this project."""
    headers = {}
    if ws_id := os.getenv("ANTHROPIC_WORKSPACE_ID"):
        headers["anthropic-workspace-id"] = ws_id
    return Anthropic(default_headers=headers)


def model_name() -> str:
    return _MODEL
