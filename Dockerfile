# ---- 构建阶段：编译 rust_core 原生扩展为 wheel ----
FROM python:3.13-slim AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin==1.14.0

WORKDIR /build
COPY rust_core/ ./rust_core/
RUN cd rust_core && maturin build --release --out /wheels

# ---- 运行阶段 ----
FROM python:3.13-slim AS runtime

# libgomp1 是 torch / sentence-transformers 的 OpenMP 运行时依赖。
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# 默认 CPU 部署：先装 CPU 版 torch，避免镜像塞入数 GB 用不上的 CUDA 包。
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY --from=rust-builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

COPY . .

# BGE 嵌入/精排模型首次使用时从 HuggingFace 下载；生产可挂载缓存卷或预下载。
# GPU 部署需换 nvidia/cuda 基础镜像并装对应 torch。
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
