FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODELS=/root/.ollama/models
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app

RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list && \
    echo 'deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse' > /etc/apt/sources.list && \
    echo 'deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse' >> /etc/apt/sources.list && \
    echo 'deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse' >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    jq \
    && rm -rf /var/lib/apt/lists/*

COPY ollama-linux-amd64.tgz /tmp/ollama.tgz
RUN tar -xzf /tmp/ollama.tgz -C /usr/local && \
    rm /tmp/ollama.tgz && \
    chmod +x /usr/local/bin/ollama

EXPOSE 11434

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

CMD ["ollama", "serve"]