#!/bin/sh
set -e

# Bridge for hosts (e.g. Hugging Face Spaces) that inject secrets as plain
# environment variables: st.secrets only reads .streamlit/secrets.toml, which
# is gitignored and never shipped in the image, so without this the app would
# see no APP_PASSWORD/ANTHROPIC_API_KEY even after the host secret is set.
if [ -n "$APP_PASSWORD" ] || [ -n "$ANTHROPIC_API_KEY" ]; then
    mkdir -p /app/.streamlit
    cat > /app/.streamlit/secrets.toml <<EOF
APP_PASSWORD = "${APP_PASSWORD}"
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"
EOF
fi

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
