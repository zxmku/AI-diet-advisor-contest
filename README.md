# 10_项目源码_healthpick · AI 智能膳食顾问源码

> 本文档为项目源码目录说明。对外使用请见 `20_交付文档/使用说明.md`（启动方式、真 AI 配置、功能清单与合规红线）。

## 目录分区

| 分区 | 职责 | 说明 |
|---|---|---|
| `frontend/` | 原生 HTML/JS 单页聊天界面（零构建、零依赖） | 含聊天输入、快捷按钮、示例气泡、来源标注、模型徽章、历史会话面板 |
| `backend/` | FastAPI 对话编排 + 推荐 + 问答 + 合规层 | 6 个 REST 端点、统一响应契约、检索/合成/成本闸门/合规层 |
| `knowledge/` | 素材 A/B/C → 结构化 JSON（人工核对后入库） | A 营养基础 16 块 / B 三套方案 14 块 / C 平台服务 15 块 + 禁忌映射 + 同义词 |
| `tests/` | pytest 全量测试 + 红线回归（R1-R8） | 环境隔离，不污染线上数据；当前 14 用例全绿 |
| `deploy/` | Dockerfile、docker-compose.yml、.env.example | 容器部署 `cd deploy && docker compose up -d` → http://localhost:8137 |

## 铁律

1. 素材原件只读：知识库 JSON 由素材加工而来，原件永远不动。
2. 真实 API Key 永不入库：只经环境变量 / `deploy/.env`（已 gitignore）注入，仓库仅保留 `.env.example` 模板。
3. 数值必须工具计算：营养数值一律来自权威速查表，禁止模型推理编造。

## 快速验证

```bash
pip install -r backend/requirements.txt
cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8137
# 浏览器打开 http://127.0.0.1:8137/
```

自动化红线回归：`python -m pytest tests -q`（14 passed）。
