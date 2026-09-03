# STATUS — Clinical Trial Intelligence System

**Portfolio project. Public data only. NOT CLINICALLY VALIDATED.**

A module is ✅ only if its code was actually run and the output inspected.
✅ Working · ⚠️ Partial · ❌ Not started · 🔴 Broken · 📄 Design-only

**18 modules** (collapsed from 23 — see `docs/modules/README.md`).

| # | Module | Status | Verified | Command | Result |
|---|---|---|---|---|---|
| 1 | Product & Problem Understanding | 📄 Done | 2026-09-03 | — | Framing, 4 use cases, 5 hard gates, success criteria |
| 2 | Architecture, Schemas & Repo Setup | ✅ Working | 2026-09-03 | `pytest tests/ -v` | 12/12 pass — HG-1..HG-4 enforced in Pydantic |
| 3 | Data Ingestion | ✅ Working | 2026-09-03 | `python -m src.ingest.verify` | All 4 live sources OK; 21/21 tests |
| 4 | Knowledge Base | ✅ Working | 2026-09-03 | `python -m src.kb.build` | 14 real docs -> 116 chunks; hybrid (FAISS+BM25) search; 5/5 sample queries relevant; 51/51 tests |
| 5 | Protocol Agent | ✅ Working | 2026-09-03 | `pytest tests/test_protocol_agent_live.py -v -s` | Real NCT04280705, grounding check verified, 27/27 tests |
| 6 | Evidence Agent | ✅ Working | 2026-09-03 | `pytest tests/test_evidence_agent_live.py -v -s` | Real PubMed query, 5 grounded claims, strength=limited; 34/34 tests |
| 7 | Regulatory Agent | ✅ Working | 2026-09-03 | `pytest tests/test_regulatory_agent_live.py -v -s` | Rewired onto Module 4's KB (was a keyword stopgap); real ICH E9 findings on interim analysis/multiplicity; 51/51 tests |
| 8 | Safety Agent | ❌ | — | — | — |
| 9 | Supervisor Agent | ❌ | — | — | — |
| 10 | Validation Layer | ❌ | — | — | — |
| 11 | Human Review Interface | ❌ | — | — | — |
| 12 | Audit Log & Observability | ❌ | — | — | — |
| 13 | Evaluation Harness | ❌ | — | — | — |
| 14 | Containerization & CI/CD | ❌ | — | — | — |
| 15 | Documentation & Demo Packaging | ❌ | — | — | — |
| 16 | Cloud Architecture & Responsible AI | ❌ | — | — | — |
| 17 | Metrics Rollup & STAR Narrative | ❌ | — | — | — |
| 18 | Mock Defense | ❌ | — | — | — |

## Blockers

| Blocker | Blocks | Needed |
|---|---|---|
| Demo drug + indication | 8, 13 | One drug with enough FAERS volume (placeholder `pembrolizumab` in use, A-06) |

## Budget

Sonnet 5, effort-tuned, Batch API for eval: **~$1.75 total** (target: under $5).
