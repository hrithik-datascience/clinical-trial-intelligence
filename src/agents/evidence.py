"""Evidence Agent (Module 6).

Given a question, searches PubMed and synthesizes an answer where every claim
is grounded in a quoted abstract passage. Same two-schema + grounding-check
pattern as Module 5's Protocol agent: the model never touches a Citation
object, and any claim whose quote isn't a literal substring of the source
abstract is rejected rather than trusted.

Per T-03 (logged before this module was written): citations are our own
chunk_id + quoted_text, not Claude's native document citations, because
`output_config.format` and native citations return a 400 when combined. The
cost of that choice is real here — see Module 6's doc, Limitations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from src.agents.errors import GroundingError, is_grounded
from src.ingest.pubmed import search_and_fetch
from src.llm import EFFORT_EXTRACTION, get_client, model_name, parse_with_retry
from src.schemas import Citation, Claim, EvidenceStrength, EvidenceSynthesis, RawDocument, SourceType


class _LlmClaim(BaseModel):
    text: str = Field(min_length=1, description="One claim, in your own words.")
    doc_index: int = Field(description="Index into the supplied document list this claim is drawn from.")
    quoted_text: str = Field(
        min_length=1,
        description="The exact substring from that document's abstract supporting the claim.",
    )


class _LlmSynthesis(BaseModel):
    strength: EvidenceStrength
    claims: list[_LlmClaim] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self) -> "_LlmSynthesis":
        if self.strength == EvidenceStrength.NONE_FOUND and self.claims:
            raise ValueError("none_found must not carry claims")
        if self.strength != EvidenceStrength.NONE_FOUND and not self.claims:
            raise ValueError("claims required unless strength=none_found")
        return self


_SYSTEM_PROMPT = """\
You synthesize evidence from published literature abstracts for a clinical \
trial reviewer. You are reporting what the literature says, not offering \
your own medical opinion.

Rules, no exceptions:
1. Every claim must be paired with a quoted_text copied VERBATIM from the \
specific document's abstract you drew it from — an exact substring, not a \
paraphrase.
2. If the retrieved abstracts do not actually address the question, or \
disagree with each other, you MUST set strength to "conflicting" or \
"limited" rather than presenting a confident synthesis. If nothing relevant \
was found at all, set strength="none_found" and return no claims.
3. Never resolve a disagreement between sources by picking the answer that \
sounds more authoritative — report the disagreement.

Weak or absent evidence, honestly reported, is a correct answer. A confident \
synthesis papering over a conflict is the failure mode this exists to avoid."""


def _build_prompt(question: str, docs: list[RawDocument]) -> str:
    listing = "\n\n".join(
        f"[{i}] {d.title}\n{d.text}" for i, d in enumerate(docs)
    )
    return f"Question: {question}\n\nRetrieved documents:\n\n{listing}"


def synthesize(question: str, max_results: int = 5, run_id: str | None = None) -> EvidenceSynthesis:
    """Search PubMed and synthesize a cited answer.

    Raises GroundingError if any claim's quote is not a literal substring of
    the document it claims to be drawn from. run_id correlates this call in
    the audit log (Module 12) with its Supervisor run; omit it outside one.
    """
    docs = search_and_fetch(question, max_results=max_results)
    docs_with_abstracts = [d for d in docs if d.metadata.get("has_abstract")]

    if not docs_with_abstracts:
        return EvidenceSynthesis(question=question, claims=[], strength=EvidenceStrength.NONE_FOUND)

    client = get_client()
    response = parse_with_retry(
        client,
        agent="evidence",
        run_id=run_id,
        model=model_name(),
        max_tokens=4000,
        output_config={"effort": EFFORT_EXTRACTION},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(question, docs_with_abstracts)}],
        output_format=_LlmSynthesis,
    )
    llm_result = response.parsed_output

    claims: list[Claim] = []
    for llm_claim in llm_result.claims:
        if not (0 <= llm_claim.doc_index < len(docs_with_abstracts)):
            raise GroundingError(f"model cited doc_index {llm_claim.doc_index}, out of range")
        doc = docs_with_abstracts[llm_claim.doc_index]
        if not is_grounded(llm_claim.quoted_text, doc.text):
            raise GroundingError(
                f"model claimed a quote not present in {doc.source_name}: "
                f"{llm_claim.quoted_text!r}"
            )
        citation = Citation(
            chunk_id=doc.doc_id,
            source_type=SourceType.LITERATURE,
            source_name=doc.source_name,
            retrieved_date=doc.retrieved_date,
            quoted_text=llm_claim.quoted_text,
        )
        claims.append(Claim(text=llm_claim.text, citations=[citation]))

    return EvidenceSynthesis(question=question, claims=claims, strength=llm_result.strength)
