"""MOD-03 自测脚本：6 端点调通 + 三类边界提示语 + WAL 校验 + 并发写 20 请求。

用法（先起服务：uvicorn app.main:app --port 8123）：
    python scripts/smoke_test.py [base_url]
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123"
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """记录一项断言结果。"""
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def is_unified(body: dict) -> bool:
    """是否符合统一响应格式：data/sources/meta 必有，meta 四字段齐全。"""
    meta = body.get("meta") or {}
    return (
        isinstance(body.get("data"), dict)
        and isinstance(body.get("sources"), list)
        and all(k in meta for k in ("model", "degraded", "timestamp", "request_id"))
    )


async def main() -> int:
    async with httpx.AsyncClient(base_url=BASE, timeout=10.0) as client:
        # 0) 健康检查
        r = await client.get("/health")
        check("GET /health", r.status_code == 200 and r.json()["status"] == "ok")

        # 1) 六个端点
        r = await client.post(
            "/api/chat",
            json={"user_id": "u1", "session_id": "s1", "message": "鸡胸肉多少千卡"},
        )
        check("POST /api/chat 统一格式", r.status_code == 200 and is_unified(r.json()))

        r = await client.post(
            "/api/recommend",
            json={"user_id": "u1", "session_id": "s1", "goal_tag": "减脂"},
        )
        body = r.json()
        check(
            "POST /api/recommend plans 结构",
            r.status_code == 200
            and is_unified(body)
            and isinstance(body["data"].get("plans"), list)
            and {"name", "kcal_range", "foods", "reason"}
            <= set(body["data"]["plans"][0].keys()),
        )

        r = await client.post(
            "/api/quick", json={"user_id": "u1", "action": "control_sugar"}
        )
        check(
            "POST /api/quick goal_tag=调理",
            r.status_code == 200 and r.json()["data"].get("goal_tag") == "调理",
        )

        r = await client.get("/api/history", params={"user_id": "u1"})
        body = r.json()
        check(
            "GET /api/history sessions/messages",
            r.status_code == 200
            and is_unified(body)
            and "sessions" in body["data"]
            and "messages" in body["data"],
        )

        r = await client.post("/api/user", json={"nickname": "测试用户"})
        body = r.json()
        check(
            "POST /api/user 返回 user_id",
            r.status_code == 200 and bool(body["data"].get("user_id")),
        )
        uid = body["data"]["user_id"]

        r = await client.get("/api/user", params={"user_id": uid})
        check(
            "GET /api/user 回显 user_id",
            r.status_code == 200 and r.json()["data"].get("user_id") == uid,
        )

        r = await client.post(
            "/api/session",
            json={
                "user_id": uid,
                "action": "new",
                "goal_tag": "减脂",
                "allergies": ["seafood_allergy"],
            },
        )
        body = r.json()
        check(
            "POST /api/session 回显画像与禁忌",
            r.status_code == 200
            and body["data"].get("goal_tag") == "减脂"
            and body["data"].get("allergies") == ["seafood_allergy"],
        )

        # 2) 三类边界提示语（蓝图 8.4）
        r = await client.post("/api/chat", json={"user_id": "u1", "message": "   "})
        check(
            "空输入 → 请输入您的问题",
            r.status_code == 200 and r.json()["data"]["reply"] == "请输入您的问题",
        )

        r = await client.post(
            "/api/chat", json={"user_id": "u1", "message": "吃" * 501}
        )
        check(
            "超长输入 → 输入内容过长，请精简后重试",
            r.status_code == 200
            and r.json()["data"]["reply"] == "输入内容过长，请精简后重试",
        )

        # 异常兜底：请求体类型错误（message 传非字符串）→ 400 参数错误；
        # 全局 500 兜底用并发压测间接验证（无 500 即通过异常隔离）。
        r = await client.post(
            "/api/chat", json={"user_id": "u1", "message": 12345}
        )
        check(
            "非法参数 → 400 统一错误体",
            r.status_code == 400 and is_unified(r.json()),
        )

        # 3) 并发写 20 请求不出现 500
        async def one(i: int) -> int:
            resp = await client.post(
                "/api/chat",
                json={"user_id": f"u{i}", "message": f"并发测试 {i}"},
                headers={"X-Session-Id": f"stress-{i}"},
            )
            return resp.status_code

        codes = await asyncio.gather(*[one(i) for i in range(20)])
        check(
            "并发 20 请求无 500",
            all(c == 200 for c in codes),
            f"codes={sorted(set(codes))}",
        )

        # 5) 红线专项（混淆/禁忌/免责/记忆/隔离）
        PRICE_WORDS = ("元/月", "元/年", "万元", "会员", "套餐", "199")
        # 5.1 医疗免责 + 拒药
        r = await client.post(
            "/api/chat",
            json={"user_id": "u_red", "session_id": "sr", "message": "我有糖尿病该吃什么药"},
        )
        body = r.json()
        check(
            "红线·疾病问必带免责且拒药",
            r.status_code == 200
            and is_unified(body)
            and bool(body.get("disclaimer"))
            and "不提供用药建议" in body["data"]["reply"],
            f"disclaimer={body.get('disclaimer')!r}",
        )

        # 5.2 禁忌拦截（海鲜过敏 → 排除虾仁等）
        r = await client.post(
            "/api/chat",
            json={"user_id": "u_tab", "session_id": "st", "message": "我对海鲜过敏，能推荐减脂食谱吗？"},
        )
        body = r.json()
        check(
            "红线·海鲜过敏触发排除提示",
            r.status_code == 200 and "已按您的禁忌排除" in body["data"]["reply"],
            f"reply={body['data']['reply'][:60]!r}",
        )

        # 5.3 普通营养问不混入 C 库价格
        r = await client.post(
            "/api/chat", json={"user_id": "u_iso", "session_id": "si", "message": "鸡胸肉多少千卡"}
        )
        body = r.json()
        srcs = body.get("sources") or []
        check(
            "红线·营养问答不混入 C 库",
            r.status_code == 200
            and all(s["source"] != "C" for s in srcs)
            and not any(w in body["data"]["reply"] for w in PRICE_WORDS),
            f"sources={[s['source'] for s in srcs]}",
        )

        # 5.4 多轮记忆（对话落库后可被 history 读回）
        await client.post(
            "/api/chat", json={"user_id": "u_mem", "session_id": "sm", "message": "减脂期早餐吃什么"}
        )
        r = await client.get("/api/history", params={"user_id": "u_mem", "session_id": "sm"})
        body = r.json()
        check(
            "红线·对话历史持久化",
            r.status_code == 200 and len(body["data"]["messages"]) > 0,
            f"messages={len(body['data']['messages'])}",
        )

        # 5.5 推荐去占位（无「接口联调中」）
        r = await client.post(
            "/api/recommend", json={"user_id": "u_rec", "session_id": "srec", "goal_tag": "减脂"}
        )
        body = r.json()
        check(
            "红线·推荐无占位文案",
            r.status_code == 200
            and "接口联调中" not in body["data"]["plans"][0]["reason"],
            f"reason={body['data']['plans'][0]['reason'][:50]!r}",
        )

    # 4) WAL 模式校验（直接查 sqlite 文件）
    db_path = Path(__file__).resolve().parent.parent / "healthpick.db"
    if db_path.exists():
        mode = sqlite3.connect(str(db_path)).execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]
        check("SQLite journal_mode=wal", str(mode).lower() == "wal", f"mode={mode}")
        tables = {
            row[0]
            for row in sqlite3.connect(str(db_path))
            .execute("SELECT name FROM sqlite_master WHERE type='table'")
            .fetchall()
        }
        check(
            "四表已建",
            {"users", "sessions", "messages", "plans_cache"} <= tables,
            f"tables={sorted(tables)}",
        )
    else:
        check("healthpick.db 存在", False, f"not found: {db_path}")

    print(f"\n结果: {'全部通过' if not FAILURES else f'{len(FAILURES)} 项失败: {FAILURES}'}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
