FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list && \
    echo 'deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse' > /etc/apt/sources.list && \
    echo 'deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse' >> /etc/apt/sources.list && \
    echo 'deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse' >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    sox \
    libsox-dev \
    libsox-fmt-all \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir pip==23.3.1 setuptools==69.0.3 wheel==0.42.0 -i https://mirrors.aliyun.com/pypi/simple/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY app.py .
COPY engines/ ./engines/
COPY config/ ./config/

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]