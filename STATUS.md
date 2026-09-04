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
| 8 | Safety Agent | ✅ Working | 2026-09-03 | `pytest tests/test_safety_agent_live.py -v -s` | Real FAERS counts (pembrolizumab) + real FDA label check, known vs. unexpected signals distinguished; 59/59 tests |
| 9 | Supervisor Agent | ✅ Working | 2026-09-03 | `pytest tests/test_supervisor_live.py -v -s` | Real intent routing + LangGraph graph; 3-agent run, 0 failures; parallelism measured; 84/84 tests |
| 10 | Validation Layer | ✅ Working | 2026-09-03 | `python -m src.pipeline` | Real packet: confidence 0.944 decomposed, 17 citations checked, 1 real cross-agent flag caught; 109/109 tests |
| 11 | Human Review Interface | ✅ Working | 2026-09-04 | `streamlit run src/ui/app.py` | Real dev server (HTTP 200), real approve/edit/reject flow, 1 live end-to-end pass; 141/141 tests |
| 12 | Audit Log & Observability | ✅ Working | 2026-09-04 | `python -c "from src import audit; ..."` | Real append-only log, 89 real events this session, real cost $0.24/14 calls; 154/154 tests |
| 13 | Evaluation Harness | ✅ Working | 2026-09-04 | `python -m src.eval.harness` | Real golden set (18 examples); Protocol P/R 1.0/1.0 after fixes; Safety agreement 1.0; 3 real bugs found, 2 fixed; 167/167 tests |
| 14 | Containerization & CI/CD | ⚠️ Partial | 2026-09-04 | `docker build -t clinical-trial-intelligence:test .` | Dockerfile, docker-compose, GitHub Actions CI, ruff lint config all written; lint passes (0 issues) and 166/167 tests pass locally; **Docker build itself unverified** — failed with "no space left on device" (host C: drive at 0 bytes free / Docker Desktop unable to start), not a code issue. Skipped by user pending disk space. |
| 15 | Documentation & Demo Packaging | ✅ Working | 2026-09-04 | — | Root `README.md` + `docs/DEMO_SCRIPT.md` written; every command cross-checked against a real file; Module 14 represented honestly as unverified, not hidden |
| 16 | Cloud Architecture & Responsible AI | ❌ | — | — | — |
| 17 | Metrics Rollup & STAR Narrative | ❌ | — | — | — |
| 18 | Mock Defense | ❌ | — | — | — |

## Blockers

| Blocker | Blocks | Needed |
|---|---|---|
| Demo drug + indication | 8, 13 | One drug with enough FAERS volume (placeholder `pembrolizumab` in use, A-06) |

## Budget

Sonnet 5, effort-tuned. Was estimated at ~$1.75 total (never measured); Module 12 now logs every
real call's actual token usage and a placeholder-priced cost (A-09) to `data/audit/audit_log.jsonl`.
Real total so far this session: **$0.24 across 14 real LLM calls** (38,131 input + 8,491 output
tokens). Target: under $5.
