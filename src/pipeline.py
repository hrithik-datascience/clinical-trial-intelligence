"""End-to-end composition: request -> Supervisor -> Validation -> BriefingPacket.

This is the one path the whole system runs on, and the human review interface
(Module 11) drives exactly this function rather than reimplementing the wiring.

Run the demo:  venv/Scripts/python.exe -m src.pipeline

The packet it returns is deliberately NOT final. Only a recorded human
decision can make it final (HG-4), which is Module 11's job.
"""

from __future__ import annotations

import sys

from src.agents.supervisor import SupervisorRequest, run
from src.kb.build import KB_DIR
from src.kb.store import KnowledgeBase
from src.schemas import BriefingPacket
from src.validation import confidence_breakdown, validate

# The demo pairing deliberately keeps ASSUMPTION A-06's placeholder drug: it
# is an oncology drug paired with a COVID antiviral trial, which is exactly
# the inconsistency the cross-agent check should catch. Leaving it in place
# means the demo exercises a real flag instead of a clean happy path.
DEMO_QUERY = (
    "Review NCT04280705 against statistical guidance, and screen the drug's "
    "reported safety signals."
)
DEMO_DRUG = "pembrolizumab"


def run_and_validate(
    request: SupervisorRequest, kb: KnowledgeBase | None = None
) -> BriefingPacket:
    return validate(run(request, kb=kb), kb=kb)


def main() -> int:
    if not (KB_DIR / "faiss.index").exists():
        print(f"No knowledge base at {KB_DIR}. Run `python -m src.kb.build` first.")
        return 1

    kb = KnowledgeBase.load(KB_DIR)
    supervisor_run = run(SupervisorRequest(query=DEMO_QUERY, drug=DEMO_DRUG), kb=kb)
    packet = validate(supervisor_run, kb=kb)

    print("Module 10 -- end-to-end run through the validation layer")
    print("=" * 74)
    print(f"\nrun_id   : {packet.run_id}")
    print(f"query    : {packet.query}")
    print(f"routing  : {[a.value for a in supervisor_run.routing.selected]} "
          f"({supervisor_run.routing.method.value})")
    print(f"produced : {[a.value for a in supervisor_run.agents_with_output]}")
    print(f"failures : {[(f.agent.value, f.error_type) for f in supervisor_run.failures]}")
    print(f"elapsed  : {supervisor_run.elapsed_seconds}s")

    print(f"\nconfidence : {packet.overall_confidence}")
    print(f"  {confidence_breakdown(supervisor_run, kb).explain()}")
    print("  (completeness/verifiability, NOT a probability of correctness -- T-21)")

    print(f"\nvalidation flags: {len(packet.validation_flags)}")
    for flag in packet.validation_flags:
        agents = "+".join(a.value for a in flag.agents_involved)
        print(f"  [{flag.flag_type.value}] ({agents})")
        print(f"      {flag.detail}")

    print(f"\nis_final : {packet.is_final}  (human_decision={packet.human_decision.value})")
    print("A packet is final only once a human has ruled on it -- HG-4, Module 11.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
