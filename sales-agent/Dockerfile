FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY sales_agent ./sales_agent
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "sales_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
