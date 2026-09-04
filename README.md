# Clinical Trial Intelligence

**Portfolio project. Public data only. NOT CLINICALLY VALIDATED.**

A governed multi-agent system for clinical protocol review. Four agents each pull one thread from
a public source — protocol structure, regulatory alignment, published literature, reported safety
signals. A validation layer cross-checks them, a confidence score is decomposed from real ratios
(never invented), and nothing is final until a human reviewer approves, edits, or rejects it. Every
step is logged to an append-only audit trail.

**The system produces a briefing. The human makes the decision.**

## The four use cases

| Use case | Agent | Source | Must never say |
|---|---|---|---|
| Regulatory pre-check | Regulatory | FDA E9, ICH guidance | "compliant" or "approved" |
| Evidence lookup | Evidence | PubMed, ClinicalTrials.gov | anything without a citation |
| Safety screening | Safety | openFDA FAERS | anything implying causality |
| Go/no-go briefing | Supervisor | all four | "proceed" or "stop" |

## Architecture

```
              query + NCT id
                    │
              Supervisor  ──── routes to the agents the query needs
                    │
      ┌─────────┬───┴────┬──────────┐
   Protocol  Evidence  Regulatory  Safety
      └─────────┴───┬────┴──────────┘
                    │  all four read from
        Shared Knowledge Base   (one index, tagged by source_type)
                    │
            Validation Layer    (deterministic checks + confidence)
                    │
          Human Review Gate     ← approve / edit / reject
                    │
       Briefing Packet + Audit Log
```

Five hard gates are enforced as Pydantic validators, not a downstream lint pass — a bad object
can't be constructed in the first place. See `docs/modules/module_02_architecture_schemas.md`.

## Status

**13 of 18 modules done and verified against real data — nothing here is invented or estimated.**
Full per-module status, verification command, and real result: [`STATUS.md`](STATUS.md).

| Area | Real result |
|---|---|
| Protocol field extraction | precision 1.0, recall 1.0 (after 2 live-found fixes) |
| Regulatory citation precision | 0.50 — traced to one known PDF-extraction artifact, not citation quality |
| Safety signal agreement | 1.0 (6/6, verified against real FDA label text) |
| End-to-end latency | 3.97s avg across 36 real Supervisor runs |
| Real measured cost | $0.24 across 14 real LLM calls |

Modules 14 (Containerization & CI/CD) is written (Dockerfile, docker-compose, GitHub Actions) but
**not yet verified end to end** — blocked on local disk space, not a code issue. Treat `docker
build` as untested until that's confirmed. Modules 16-18 (cloud architecture design, metrics
rollup, mock defense) are documentation/interview work, not code, and don't depend on 14.

## Setup

```bash
python -m venv venv
venv/Scripts/activate          # venv\Scripts\activate.bat on Windows cmd
pip install -r requirements.txt
cp .env.example .env           # fill in ANTHROPIC_API_KEY
```

## Running it

```bash
# 1. Verify all four live data sources are reachable
python -m src.ingest.verify

# 2. Build the shared knowledge base (ClinicalTrials.gov, ICH/FDA guidance, PubMed, FAERS)
python -m src.kb.build

# 3. Launch the review UI
streamlit run src/ui/app.py

# Run the test suite (166/167 passing — the one failure is a live-API grounding
# test hitting real PubMed/Claude output, a documented flake, not a code defect)
venv/Scripts/python.exe -m pytest tests/ -v

# Run the evaluation harness against the golden set
venv/Scripts/python.exe -m src.eval.harness
```

A step-by-step walkthrough of all four use cases is in
[`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).

## Docs

| File | Purpose |
|---|---|
| [`STATUS.md`](STATUS.md) | What actually runs, per module, with real verification |
| [`docs/modules/`](docs/modules/) | One file per module — what was built, real findings, limitations |
| [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) | Current state, CAN/CANNOT-say claims |
| [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md) | Project-level decisions with alternatives |
| [`docs/TECHNIQUE_METHOD_LOG.md`](docs/TECHNIQUE_METHOD_LOG.md) | Method choices with alternatives, per technique |
| [`docs/ASSUMPTION_REGISTER.md`](docs/ASSUMPTION_REGISTER.md) | Assumptions still open |
| [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) | Live walkthrough script + recording plan |

## Limitations (read before demoing)

1. Public registry listings aren't real draft protocols — extraction accuracy won't fully transfer.
2. FAERS shows reported associations only, never causality — enforced in code (HG-3), not just prose.
3. The golden set (18 examples) is developer-hand-labeled; the user confirmed the reported results
   acceptable (2026-09-04), but no independent clinical/regulatory reviewer has checked the labels.
4. `cost_usd` in the audit log is a placeholder price conversion applied to real, measured token counts.
5. Not deployed, not containerized end-to-end (yet), not validated for any real clinical or
   regulatory decision — at no point does this project claim otherwise.
