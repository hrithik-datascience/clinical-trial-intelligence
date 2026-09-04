#!/bin/sh
set -e

# Module 14 (T-28): the knowledge base is built here, at container START,
# never at `docker build` time -- CI runners and locked-down build
# environments commonly restrict network access, and building the KB needs
# four live public APIs (ClinicalTrials.gov, FDA/ICH, PubMed, openFDA).
# This mirrors src/agents/supervisor.py::_load_kb(), which raises the same
# "run the build first" message when called directly rather than a UI.
if [ ! -f "/app/data/index/faiss.index" ]; then
    echo "No knowledge base found at /app/data/index -- building it now"
    echo "(real ingestion from 4 live public sources + local embeddings, one-time)..."
    python -m src.kb.build
fi

exec "$@"
