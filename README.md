# AI 智能膳食顾问（HealthPick）· v1.20 最终版本

> 本仓库为最终交付版本：初版因误以为赛题按"48 小时"截止先行提交至平台，后发现平台实际截止为 **8 月 26 日**，故在窗口期内持续迭代完善。**请以本仓库（GitHub）为最终版本评审**，全部能力、文档与测试均以本仓库为准。版本故事详见《版本说明.md》（位于 `项目资料/`）。
>
> ⚠️ **本仓库仅提供本地部署，没有云端 / 公网地址。** 下面的方式都在你自己的电脑上跑，不依赖任何外网服务器，访问地址统一为 `http://localhost:8137`。

---

## 产品是什么

面向「减脂塑形 / 增肌强化 / 慢病调理」人群的 **AI 膳食陪伴顾问**：不止查热量、推方案，更能**搭出完整一餐**（食材 + 克数 + 热量 + 做法 + 替换）、**记住你的目标与过敏**、**在你加班嘴馋时先共情再给无痛替代**——聊起来、记得我、陪你坚持。

- **知识库问答**：素材 A 营养基础 / B 三套方案 / C 平台服务（物理隔离），回答带来源标注、数值 100% 查表不编造
- **多轮对话与记忆**：会话上下文 + 目标/过敏画像跨轮延续 + 饮食台账与坚持天数
- **方案与一餐**：减脂 / 增肌 / 控糖三套方案 + 一餐生成（速查表 × 份量计算）
- **红线合规**：拒药硬拦截 / 禁忌全链路排除 / 医疗免责 / C 库隔离 / AI 辅助标注如实
- **双模式**：配置密钥 = LLM 产品级；未配置 = 本地规则兜底（功能完整，仅无 AI 润色）

---

## 🖥️ 本地部署方法（三种，按省事程度排）

### 方式一 · Docker 部署（最省心，推荐）

前提：本机装了 Docker Desktop。

1. 进 `deploy` 目录，把 `.env.example` 复制成 `.env`（不填也能跑，只是没有 AI 润色）。
2. 在项目根目录执行：
   ```bash
   docker compose -f deploy/docker-compose.yml up -d
   ```
3. 浏览器打开 http://localhost:8137 。

镜像里已经把后端、知识库、前端全打进去了。不填 Key 是纯本地规则模式，填了 `DEEPSEEK_API_KEY` 就升级成 LLM 动态生成，核心硬功能（数值查表、禁忌排除、拒药）两种模式都在。

### 方式二 · Windows 离线双击（零联网）

适合没装 Docker、也不想配环境的本机。

1. 双击 `deploy/一键启动.bat`。
2. 脚本会自动找本机 Python 3.10 / 3.11 / 3.12，从仓库自带的 `deploy/wheels` 离线装依赖，起好服务后自动开浏览器（端口 8137）。
3. 想要真 AI：用记事本打开 `deploy/.env`，填 `DEEPSEEK_API_KEY=你的密钥`，重双击一次。

整个过程不访问外网，依赖包都随仓库带着，干净 Windows 双击就有反应。

### 方式三 · 手动 Python 跑

1. 用 Python 3.10–3.12，建虚拟环境：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. 装依赖：
   ```bash
   pip install -r backend/requirements.txt
   ```
   离线环境改用：`pip install --no-index --find-links deploy/wheels -r backend/requirements.txt`
3. 配密钥（可选）：复制 `deploy/.env.example` 为 `backend/.env`，填 `DEEPSEEK_API_KEY`。
4. 起服务：
   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8137
   ```
5. 打开 http://localhost:8137 ，或者先打 http://localhost:8137/health 看是不是返回 `ok`。

### 环境变量（参考 deploy/.env.example）

- `HEALTHPICK_ENV`：运行环境，`dev` 或 `prod`
- `HEALTHPICK_PORT`：监听端口，默认 8137
- `DEEPSEEK_API_KEY`：留空 = 本地规则降级；填了 = LLM 模式
- `DATABASE_URL`：默认本地 SQLite，一般不填
- `HEALTHPICK_RATE_LIMIT_PER_MIN`：每分钟限速，默认 60

### 大模型 Key 从哪来

去 DeepSeek 开放平台申请一个 Key，填进 `.env` 的 `DEEPSEEK_API_KEY` 就行。不填也能完整用，只是少了 AI 那层润色；数值查表、禁忌排除、拒药拦截这些核心硬功能不受影响。

### 健康检查

服务起来后访问 `/health`，返回 `{"status":"ok", ...}` 就是正常的。

---

## 怎么看这个仓库（评审入口）

| 你想看 | 去哪 |
|---|---|
| **提交文档**（交付物总览、人设/架构出处、AI 辅助标注） | [`提交文档.md`](提交文档.md) |
| **本地部署方法**（本文已含；另见分步详解） | [`本地部署方法.md`](本地部署方法.md) |
| 需求拆解、业务 SOP、架构蓝图、交付物详情、基线地图、测试记录、版本说明、复盘 | [`项目资料/`](项目资料/) |
| 后端 / 前端 / 知识库 / 测试 / 部署脚本 | 见下方「工程目录」 |

## 工程目录

| 目录 | 职责 |
|---|---|
| `backend/` | FastAPI 后端（对话编排 / 合规层 / 检索 / 一餐生成 / 台账） |
| `frontend/` | 单页聊天界面（零构建、零依赖） |
| `knowledge/` | 素材 A/B/C 结构化知识库 + 禁忌映射 + 同义词 |
| `tests/` | pytest 全量测试（详见 `项目资料/30_测试记录/`） |
| `deploy/` | Dockerfile / docker-compose / .env.example / 一键启动脚本 / 离线 wheels |
| `assets/` | 静态资源 |
| `项目资料/` | 需求/架构/交付/测试/版本/复盘等背景资料（见其内 `README` 或各子目录） |

> 代码与交付物在顶层清晰分区；需求拆解、业务 SOP、架构蓝图、各类测试记录等**背景资料统一收口在 `项目资料/`**，不占顶层视线。

## 质量门禁

自动化测试：`python -m pytest tests -q` 全量通过。含红线回归、100 条攻击式指令压力测试（暴风雪）、外部审查与用户视角刁难压测，缺陷闭环并回归锁死；详见 `项目资料/30_测试记录/`。

## 技术要点

- FastAPI 单端口托管前后端，前端零构建
- BM25 检索（rank_bm25 + jieba + 同义词扩展），A/B 库合并索引、C 库独立隔离
- LLM 合成层（DeepSeek）：检索 grounding + 用户画像注入 + 成本闸门 / 限速 / 缓存
- 双引擎容灾：未配置密钥或模型失败 → 自动本地规则兜底，绝不白屏
- 仅本地部署：服务监听本机 `http://localhost:8137`，无公网 / 云端地址

---

完整源代码、工程结构、文档与测试托管于 GitHub：
**https://github.com/zxmku/AI-diet-advisor-contest**
