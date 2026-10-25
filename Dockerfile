FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py auth.py codex.py ./
COPY static ./static
ENV HOST=0.0.0.0 \
    TOKEN_STORE=/data/tokens.json
VOLUME /data
EXPOSE 8000 1455
CMD ["python", "server.py"]
