"""Errors and the grounding check shared across agents.

Split out during Module 6: GroundingError was originally defined inside
protocol.py, and evidence.py needed the same check. Agents importing from
each other is a smell — a shared errors module isn't.

DEVIATION FROM PLAN, found running Module 6 against real PubMed abstracts:
the first live run raised a false-positive GroundingError. The source
abstract used Unicode thin spaces around "=" (U+2009: "CI = 9.2"),
NCBI's real typographic formatting, and the model faithfully quoted the same
content with ordinary ASCII spaces — a correct quote, rejected by a naive
`substring in text` check. Fixed by normalizing whitespace (collapsing any
run of Unicode whitespace to a single ASCII space) on both sides before
comparing. This does not weaken the check against actual fabrication — an
invented quote does not become real by collapsing its spaces — it only stops
penalizing the model for something no LLM can be expected to reproduce
byte-for-byte: invisible formatting.
"""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def is_grounded(quoted_text: str, source_text: str) -> bool:
    """True if quoted_text appears in source_text, ignoring whitespace
    differences (including Unicode thin/non-breaking spaces)."""
    return _normalize(quoted_text) in _normalize(source_text)


class GroundingError(ValueError):
    """An agent's model call claimed a quote that is not actually present in
    the source document it cited. Raised, not silently downgraded — a
    fabricated quote is a worse failure than an honest abstention.
    """
