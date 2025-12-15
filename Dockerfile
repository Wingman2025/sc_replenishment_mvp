FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY sc_replenishment_mvp/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install -r /app/requirements.txt

COPY sc_replenishment_mvp /app/sc_replenishment_mvp

EXPOSE 8080

CMD ["sh", "-c", "python -m streamlit run sc_replenishment_mvp/app.py --server.address 0.0.0.0 --server.port ${PORT:-8080} --server.headless true --server.showEmailPrompt false --browser.gatherUsageStats false"]
