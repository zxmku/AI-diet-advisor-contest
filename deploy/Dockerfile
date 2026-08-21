# HealthPick 后端镜像：零配置可跑（纯规则降级），配 Key 即升级为 LLM 模式。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HEALTHPICK_ENV=prod \
    HEALTHPICK_HOST=0.0.0.0 \
    HEALTHPICK_PORT=8000

WORKDIR /app

# 依赖先装（利用层缓存）
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端与知识库（知识库为本地 JSON，无需外网）
COPY backend/ ./backend/
COPY knowledge/ ./knowledge/

WORKDIR /app/backend
EXPOSE 8000

# 启动即建表 + 预热检索索引；prod 下关闭 /docs 防接口地图泄露
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"]
