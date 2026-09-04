"""Evaluation Harness (Module 13).

Runs the golden set (src/eval/golden_set.py, T-25) against the real agents
and computes exactly the metrics the master prompt asks for: Protocol
field-extraction precision/recall, Regulatory citation precision, Safety
signal agreement rate, end-to-end latency, human-override rate. Every
number below is computed from an actual run against real data — Section H
forbids inventing one, and this module exists to make that checkable.

Run:  venv/Scripts/python.exe -m src.eval.harness
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from src import audit
from src.agents import protocol, regulatory
from src.agents.safety import is_known_label_risk
from src.eval.golden_set import PROTOCOL_EXAMPLES, REGULATORY_EXAMPLES, SAFETY_EXAMPLES
from src.ingest.openfda import fetch_label
from src.kb.build import KB_DIR
from src.kb.store import KnowledgeBase
from src.schemas import AuditEventType, FieldStatus, ReviewDecision

# --------------------------------------------------------------------------
# Protocol: field-extraction precision/recall
# --------------------------------------------------------------------------


@dataclass
class ProtocolEvalResult:
    field_confusion: dict[str, dict[str, int]]
    value_mismatches: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        tp = sum(c["TP"] for c in self.field_confusion.values())
        fp = sum(c["FP"] for c in self.field_confusion.values())
        return tp / (tp + fp) if (tp + fp) else None

    @property
    def recall(self) -> float | None:
        tp = sum(c["TP"] for c in self.field_confusion.values())
        fn = sum(c["FN"] for c in self.field_confusion.values())
        return tp / (tp + fn) if (tp + fn) else None


def eval_protocol() -> ProtocolEvalResult:
    from src.eval.golden_set import PROTOCOL_FIELD_NAMES

    confusion = {name: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for name in PROTOCOL_FIELD_NAMES}
    result = ProtocolEvalResult(field_confusion=confusion)

    for example in PROTOCOL_EXAMPLES:
        try:
            extraction = protocol.extract(example.nct_id)
        except Exception as exc:
            result.errors.append(f"{example.nct_id}: {type(exc).__name__}: {exc}")
            continue

        for name, expectation in example.fields.items():
            actual = getattr(extraction, name)
            extracted = actual.status is FieldStatus.EXTRACTED

            if expectation.should_extract and extracted:
                confusion[name]["TP"] += 1
                if expectation.value_contains and expectation.value_contains.lower() not in (actual.value or "").lower():
                    result.value_mismatches.append(
                        f"{example.nct_id}.{name}: expected value containing "
                        f"{expectation.value_contains!r}, got {actual.value!r}"
                    )
            elif expectation.should_extract and not extracted:
                confusion[name]["FN"] += 1
            elif not expectation.should_extract and extracted:
                confusion[name]["FP"] += 1
            else:
                confusion[name]["TN"] += 1

    return result


# --------------------------------------------------------------------------
# Regulatory: citation precision + expected-clause detection
# --------------------------------------------------------------------------


@dataclass
class RegulatoryEvalResult:
    matched_findings: int = 0
    total_findings: int = 0
    examples_with_expected: int = 0
    examples_with_expected_hit: int = 0
    over_flag_violations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def citation_precision(self) -> float | None:
        return self.matched_findings / self.total_findings if self.total_findings else None

    @property
    def expected_clause_detection_rate(self) -> float | None:
        return (
            self.examples_with_expected_hit / self.examples_with_expected
            if self.examples_with_expected else None
        )


def eval_regulatory(kb: KnowledgeBase) -> RegulatoryEvalResult:
    result = RegulatoryEvalResult()

    for example in REGULATORY_EXAMPLES:
        try:
            findings = regulatory.review(example.protocol_summary, kb)
        except Exception as exc:
            result.errors.append(f"{example.id}: {type(exc).__name__}: {exc}")
            continue

        result.total_findings += len(findings)
        result.matched_findings += sum(1 for f in findings if f.clause_id in example.expected_clause_ids)

        if example.expected_clause_ids:
            result.examples_with_expected += 1
            if any(f.clause_id in example.expected_clause_ids for f in findings):
                result.examples_with_expected_hit += 1

        if example.max_expected_findings is not None and len(findings) > example.max_expected_findings:
            result.over_flag_violations.append(
                f"{example.id}: {len(findings)} findings > max {example.max_expected_findings}"
            )

    return result


# --------------------------------------------------------------------------
# Safety: signal agreement rate -- no LLM call at all, pure classification
# check against independently-read label text (T-25)
# --------------------------------------------------------------------------


@dataclass
class SafetyEvalResult:
    correct: int = 0
    total: int = 0
    mismatches: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float | None:
        return self.correct / self.total if self.total else None


def eval_safety() -> SafetyEvalResult:
    result = SafetyEvalResult()
    label_cache: dict[str, str] = {}

    for example in SAFETY_EXAMPLES:
        try:
            if example.drug not in label_cache:
                label_cache[example.drug] = fetch_label(example.drug).reference_text
            actual = is_known_label_risk(example.reaction_term, label_cache[example.drug])
        except Exception as exc:
            result.errors.append(f"{example.drug}/{example.reaction_term}: {type(exc).__name__}: {exc}")
            continue

        result.total += 1
        if actual == example.expected_known_label_risk:
            result.correct += 1
        else:
            result.mismatches.append(
                f"{example.drug}/{example.reaction_term}: expected "
                f"{example.expected_known_label_risk}, got {actual}"
            )

    return result


# --------------------------------------------------------------------------
# End-to-end latency + human-override rate -- read from the real audit log
# (Module 12), not re-derived. Both are honestly caveated on small/
# non-representative sample sizes rather than presented as stable rates.
# --------------------------------------------------------------------------


def eval_latency_and_overrides() -> dict:
    events = audit.read_events()
    runs = [e for e in events if e.event_type is AuditEventType.RUN_COMPLETED]
    decisions = [e for e in events if e.event_type is AuditEventType.HUMAN_DECISION]

    latencies = [e.payload["elapsed_seconds"] for e in runs]
    overrides = sum(1 for e in decisions if e.payload["human_decision"] != ReviewDecision.APPROVED.value)

    return {
        "n_runs": len(latencies),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "n_human_decisions": len(decisions),
        "human_override_rate": round(overrides / len(decisions), 3) if decisions else None,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def main() -> int:
    if not (KB_DIR / "faiss.index").exists():
        print(f"No knowledge base at {KB_DIR}. Run `python -m src.kb.build` first.")
        return 1
    kb = KnowledgeBase.load(KB_DIR)

    print("Module 13 -- Evaluation Harness")
    print("=" * 78)
    print(f"Golden set: {len(PROTOCOL_EXAMPLES)} protocol, {len(REGULATORY_EXAMPLES)} regulatory, "
          f"{len(SAFETY_EXAMPLES)} safety examples")
    print("NOTE: hand-labeled by the developer, not yet confirmed by the user (T-25).")

    print("\n--- Protocol field-extraction ---")
    started = time.monotonic()
    p = eval_protocol()
    print(f"  ({round(time.monotonic() - started, 1)}s, {len(PROTOCOL_EXAMPLES)} real extractions)")
    for name, c in p.field_confusion.items():
        print(f"  {name:<20} TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']}")
    print(f"  precision={p.precision}  recall={p.recall}")
    if p.value_mismatches:
        print(f"  value mismatches ({len(p.value_mismatches)}):")
        for m in p.value_mismatches:
            print(f"    - {m}")
    if p.errors:
        print(f"  ERRORS: {p.errors}")

    print("\n--- Regulatory citation precision ---")
    started = time.monotonic()
    r = eval_regulatory(kb)
    print(f"  ({round(time.monotonic() - started, 1)}s, {len(REGULATORY_EXAMPLES)} real reviews)")
    print(f"  citation_precision={r.citation_precision} ({r.matched_findings}/{r.total_findings} findings matched)")
    print(f"  expected_clause_detection_rate={r.expected_clause_detection_rate} "
          f"({r.examples_with_expected_hit}/{r.examples_with_expected})")
    if r.over_flag_violations:
        print(f"  over-flag violations: {r.over_flag_violations}")
    if r.errors:
        print(f"  ERRORS: {r.errors}")

    print("\n--- Safety signal agreement rate ---")
    s = eval_safety()
    print(f"  agreement_rate={s.agreement_rate} ({s.correct}/{s.total})")
    if s.mismatches:
        print(f"  mismatches: {s.mismatches}")
    if s.errors:
        print(f"  ERRORS: {s.errors}")

    print("\n--- End-to-end latency + human-override rate (from the real audit log) ---")
    lo = eval_latency_and_overrides()
    print(f"  {lo['n_runs']} real Supervisor runs logged -> avg latency {lo['avg_latency_s']}s")
    print(f"  {lo['n_human_decisions']} real human decisions logged -> override rate {lo['human_override_rate']}")
    print("  CAVEAT: these decisions are this session's own test/demo clicks, not real clinical")
    print("  reviewer judgment -- reported honestly as a sample size, not a stable production rate.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
