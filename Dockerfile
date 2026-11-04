# ============ 构建阶段：编译依赖(asyncmy 需要 gcc) ============
FROM python:3.13-slim AS builder

# 基础镜像 /tmp 被改成 0755 时 apt 降权到 _apt 用户会无法创建临时文件
# 报 Unable to mkstemp /tmp/apt.sig.xxx - GetTempFile (13: Permission denied)
RUN chmod 1777 /tmp

WORKDIR /app

# asyncmy 无预编译 wheel 时需要编译工具链
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        default-libmysqlclient-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 把依赖装进独立目录 方便下一阶段整体拷贝
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============ 运行阶段：只带运行时 镜像更小 ============
FROM python:3.13-slim

# 运行阶段以 appuser 运行 media_evidence 会在 /tmp 建临时目录 同样需要 1777
RUN chmod 1777 /tmp

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=qa \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 从构建阶段拷贝已安装的依赖
COPY --from=builder /install /usr/local

# 拷贝项目代码(.dockerignore 已排除 venv/.env 等)
COPY . .

# 非 root 用户运行 更安全
RUN useradd --no-create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8022

# 健康检查(含数据库探活) 启动时要先同步 OSS 规则 故 start-period 放宽
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8022/health', timeout=8)"

# 启动前先同步 OSS 打分规则，再启动服务
CMD ["sh", "-c", "PYTHONPATH=. python utils/extract_score_rules.py && uvicorn main:app --host 0.0.0.0 --port 8022"]
