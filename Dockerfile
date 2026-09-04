# Module 14. Single-stage, python:3.11-slim (T-27) -- sentence-transformers'
# PyTorch dependency dominates image size regardless of stage count, so a
# multi-stage split buys single-digit MB against a multi-GB image. Not
# worth the added complexity at this project's scale.
FROM python:3.11-slim

WORKDIR /app

# curl is needed for the HEALTHCHECK below (Streamlit's own health endpoint).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN chmod +x ./docker/entrypoint.sh

# data/ (knowledge base index + audit log) is written at `docker run` time,
# never baked into the image (T-28) -- mount it as a volume so it survives
# a container restart instead of being rebuilt from four live APIs every time.
VOLUME ["/app/data"]

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
