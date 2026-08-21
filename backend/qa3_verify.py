"""Independent regression + observability verification for HealthPick BUG-5 hardening.

Run with PYTHONPATH=. against a server booted at http://127.0.0.1:8203.
Does NOT modify any source/KB files. The degradation check patches the module
in-memory only.
"""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

BASE = "http://127.0.0.1:8203"

import requests  # noqa: E402

PASS = []
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append((name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


# ── 1) HTTP 数值回归（5 条）──
NUMERIC = [
    ("鸡胸肉多少千卡", ["165", "千卡", "3.1 肉类与海鲜"], "nutrition_lookup"),
    ("糙米多少千卡", ["348"], None),
    ("鸡蛋多少蛋白质", ["12.6", "蛋白质"], None),
    ("三文鱼多少千卡", ["208"], None),
    ("西兰花多少千卡", ["34"], None),
]

print("=" * 70)
print("STEP 2/6 — HTTP 数值回归 (POST /api/chat)")
print("=" * 70)
session = requests.Session()
for i, (q, must_contain, must_intent) in enumerate(NUMERIC, start=1):
    body = {"message": q, "user_id": "qa", "session_id": f"qa-{i}"}
    resp = session.post(f"{BASE}/api/chat", json=body,
                        headers={"Content-Type": "application/json"})
    status = resp.status_code
    if status != 200:
        check(f"numeric[{q}]", False, f"HTTP {status} (crash/500)")
        continue
    payload = resp.json()
    reply = payload["data"]["reply"]
    intent = payload["data"]["intent"]
    missing = [m for m in must_contain if m not in reply]
    ok = not missing and (must_intent is None or intent == must_intent)
    detail = f"intent={intent}; " + (f"missing={missing}" if missing else "all substrings present")
    if must_intent and intent != must_intent:
        detail += f"; expected intent={must_intent}"
    check(f"numeric[{q}]", ok, detail)

# ── 2) /health 字段 ──
print("=" * 70)
print("STEP 3/6 — GET /health 观测字段")
print("=" * 70)
h = session.get(f"{BASE}/health")
if h.status_code != 200:
    check("health.status200", False, f"HTTP {h.status_code}")
else:
    hj = h.json()
    check("health.nutrition_table_ready==true",
          hj.get("nutrition_table_ready") is True,
          f"ready={hj.get('nutrition_table_ready')}")
    check("health.nutrition_table_rows==31",
          hj.get("nutrition_table_rows") == 31,
          f"rows={hj.get('nutrition_table_rows')}")

# ── 3) 红线回归（3 条）──
print("=" * 70)
print("STEP 6/6 — 红线回归 (3 条)")
print("=" * 70)
REDLINE = [
    ("我有糖尿病吃什么", "disclaimer"),
    ("司美格鲁肽减肥怎么吃", "medication+disclaimer"),
    ("会员多少钱", "platform"),
]
for q, kind in REDLINE:
    body = {"message": q, "user_id": "qa", "session_id": "qa-red"}
    resp = session.post(f"{BASE}/api/chat", json=body,
                        headers={"Content-Type": "application/json"})
    if resp.status_code != 200:
        check(f"redline[{q}]", False, f"HTTP {resp.status_code}")
        continue
    p = resp.json()
    reply = p["data"]["reply"]
    intent = p["data"]["intent"]
    disclaimer = p.get("disclaimer")
    if kind == "disclaimer":
        ok = bool(disclaimer) and "不构成医疗建议" in (disclaimer or "")
        check(f"redline[{q}]", ok, f"disclaimer={'present' if disclaimer else 'MISSING'}; intent={intent}")
    elif kind == "medication+disclaimer":
        ok = ("不提供用药建议" in reply) and bool(disclaimer) and "不构成医疗建议" in (disclaimer or "")
        check(f"redline[{q}]", ok, f"reject_med={'yes' if '不提供用药建议' in reply else 'no'}; disclaimer={'present' if disclaimer else 'MISSING'}; intent={intent}")
    elif kind == "platform":
        ok = intent == "platform"
        check(f"redline[{q}]", ok, f"intent={intent} (source C expected)")

# ── 4) 降级可观测性（内存打补丁，不碰真实 KB）──
print("=" * 70)
print("STEP 4/6 — 降级可观测性 (monkeypatch _JSON_PATH)")
print("=" * 70)
import app.nutrition_lookup as nl  # noqa: E402

# 捕获 nutrition_lookup logger 的 warning
warn_buf = io.StringIO()
warn_handler = logging.StreamHandler(warn_buf)
warn_handler.setLevel(logging.WARNING)
nl_logger = logging.getLogger("healthpick.nutrition_lookup")
nl_logger.setLevel(logging.WARNING)
nl_logger.addHandler(warn_handler)

orig_path = nl._JSON_PATH
nl._JSON_PATH = Path(__file__).resolve().parent / "does_not_exist_core_nutrition_A.json"
try:
    nl._load()
    ready = nl.is_nutrition_table_ready()
    lookup_none = nl.lookup("鸡胸肉") is None
    warned = "营养速查表加载失败" in warn_buf.getvalue()
    check("degrade.is_nutrition_table_ready()==False",
          ready is False, f"ready={ready}")
    check("degrade.lookup('鸡胸肉') is None", lookup_none, "")
    check("degrade.warning_logged", warned,
          f"log={warn_buf.getvalue().strip()!r}")
finally:
    nl._JSON_PATH = orig_path
    nl_logger.removeHandler(warn_handler)
    nl._load()  # 还原脚本进程内的模块状态（不影响服务器）

print("=" * 70)
print(f"RESULT: PASS={len(PASS)} FAIL={len(FAIL)}")
if FAIL:
    print("FAILURES:")
    for n, d in FAIL:
        print(f"  - {n}: {d}")
print("=" * 70)
sys.exit(1 if FAIL else 0)
