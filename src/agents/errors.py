"""Errors and the grounding check shared across agents.

Split out during Module 6: GroundingError was originally defined inside
protocol.py, and evidence.py needed the same check. Agents importing from
each other is a smell -- a shared errors module isn't.

DEVIATION FROM PLAN, found running Module 6 against real PubMed abstracts, in
two rounds:

1. First live run: false-positive GroundingError. The source abstract used
   Unicode thin spaces around "=" (U+2009: "CI = 9.2"), NCBI's real
   typographic formatting, and the model faithfully quoted the same content
   with ordinary ASCII spaces. Fixed by normalizing whitespace (collapsing
   any run of Unicode whitespace to a single ASCII space).

2. Second live run, same PMID, a different false positive: the source read
   "...group and 5.7 months...the median PFS was 11.3 months..." (lowercase
   "the", a mid-sentence fragment); the model quoted it as "The median PFS
   was 11.3 months..." -- capitalized, because it presented the quote as a
   standalone sentence. Content was identical; only the first letter's case
   differed. Fixed by also case-folding both sides before comparing.

Neither fix weakens the check against actual fabrication -- an invented
quote does not become real by collapsing its spaces or its case -- they only
stop penalizing the model for presentational normalization no LLM reliably
avoids when asked to quote a "sentence" out of running text.

DEVIATION FROM PLAN, found running Module 13's evaluation harness against a
wider golden set than any single agent module had exercised before:
GroundingError on real ClinicalTrials.gov eligibility text quoting
"PEFR \\> 50%". ClinicalTrials.gov's API returns "<" and ">" inside
eligibility criteria text as literal backslash-escaped Markdown ("\\<",
"\\>") -- verified across four real trials, present in 2 of 4, including
NCT04280705, this project's very first example, which had simply never
happened to trigger it before. The model naturally reads "\\>" as ">" and
quotes it unescaped -- semantically identical content, byte-for-byte
different. Fixed by stripping the backslash immediately before "<" or ">"
before comparing, the same principle as the whitespace/case fixes: this
only forgives a presentational escaping artifact, verified it still rejects
a genuinely different quote.
"""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")
_ESCAPED_ANGLE_BRACKET = re.compile(r"\\([<>])")


def _normalize(text: str) -> str:
    text = _ESCAPED_ANGLE_BRACKET.sub(r"\1", text)
    return _WHITESPACE.sub(" ", text).strip().casefold()


def is_grounded(quoted_text: str, source_text: str) -> bool:
    """True if quoted_text appears in source_text, ignoring whitespace
    differences (including Unicode thin/non-breaking spaces) and case."""
    return _normalize(quoted_text) in _normalize(source_text)


class GroundingError(ValueError):
    """An agent's model call claimed a quote that is not actually present in
    the source document it cited. Raised, not silently downgraded -- a
    fabricated quote is a worse failure than an honest abstention.
    """
