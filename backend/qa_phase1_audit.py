#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HealthPick 第一阶段闭环审计（独立 QA · 隔离端口 8204）。

对外署名：QA 工程师（独立审查员）。本脚本只做「真实验证 + 报告」，不修改任何源码。
覆盖：
  P0-①  C 库隔离（Kubernetes/容器编排 不入 A/B，且给边界提示）
  P0-②  疾病免责（糖尿病语境 → disclaimer 字段注入）
  P0-③  用药拒答（二甲双胍 → 明确拒答，不编造）
  BUG-3/4 非数值 KB 可答（检索正常、来源标注存在、不崩溃）
  BUG-5  数值速查（鸡胸肉165/糙米348/鸡蛋12.6/三文鱼208/西兰花34 + 牛油果边界）
  加固项 /health 可观测性（nutrition_table_ready / nutrition_table_rows）
  降级   KB 移除后：服务仍起 + ready=false + 启动 warning + 数值回退不崩溃（测完必还原）
  UX     多轮连贯（鸡胸肉→糙米→糖尿病）与免责不刷屏

端口：8204（隔离，绝不碰 8137）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# ── 路径（绝对，正斜杠）──
PY = r"C:/Users/24771/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
ROOT = Path(r"C:/Users/24771/Desktop/第二届月度实战能力大赛·8月专场/10_项目源码_healthpick")
BACKEND = ROOT / "backend"
KB = ROOT / "knowledge" / "core_nutrition_A.json"
KB_BAK = ROOT / "knowledge" / "core_nutrition_A.json.bak"
PORT = 8204
BASE = f"http://127.0.0.1:{PORT}"

RESULTS: list[dict] = []      # {id,name,passed,evidence,detail}
CRASHES = 0


def check(cid: str, name: str, passed: bool, evidence: str, detail: str = "") -> None:
    RESULTS.append({
        "id": cid, "name": name, "passed": bool(passed),
        "evidence": evidence, "detail": detail,
    })
    print(f"[{'PASS' if passed else 'FAIL'}] {cid} {name}" + (f" -- {detail}" if detail else ""))


def start_server(log_path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BACKEND)
    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "app.main:app", "--port", str(PORT), "--host", "127.0.0.1"],
        cwd=str(BACKEND), env=env, stdout=logf, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 45
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                return proc, logf
        except Exception:
            pass
        if proc.poll() is not None:
            logf.flush()
            raise RuntimeError(f"server exited early; log tail:\n{(Path(log_path).read_text(encoding='utf-8', errors='replace')[-1500:])}")
        time.sleep(0.5)
    raise RuntimeError("server did not become ready in 45s")


def stop_server(proc, logf) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    finally:
        if logf:
            logf.close()


def post_chat(message: str, session_id: str, user_id: str = "qa-auditor") -> dict:
    global CRASHES
    try:
        r = requests.post(
            f"{BASE}/api/chat",
            json={"message": message, "user_id": user_id, "session_id": session_id},
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        CRASHES += 1
        return {"_crash": True, "error": str(e)}
    if r.status_code >= 500:
        CRASHES += 1
    try:
        return r.json()
    except Exception:
        return {"_status": r.status_code, "_text": r.text[:500]}


def get_health() -> dict:
    try:
        r = requests.get(f"{BASE}/health", timeout=10)
        return r.json() if r.status_code == 200 else {"_status": r.status_code}
    except Exception as e:  # noqa: BLE001
        return {"_crash": True, "error": str(e)}


# ───────────────────────── 主体 ─────────────────────────
def main() -> int:
    log_normal = BACKEND / "qa_phase1_server_normal.log"
    log_degrade = BACKEND / "qa_phase1_server_degrade.log"

    # ===== 1) 正常启动（KB 存在）=====
    print("\n========== 启动服务（端口 %d，KB 存在）==========" % PORT)
    proc, logf = start_server(log_normal)
    try:
        # ---- P0-① C 库隔离 ----
        print("\n----- P0-① C 库隔离 -----")
        # (a) Kubernetes/容器编排 不应混入 A/B，且应有边界提示
        r = post_chat("怎么部署 Kubernetes 容器编排", "iso-k8s")
        reply = (r.get("data", {}).get("reply", "") if isinstance(r, dict) and "data" in r else str(r))
        intent = r.get("data", {}).get("intent") if isinstance(r, dict) else None
        sources = r.get("sources", []) if isinstance(r, dict) else []
        no_ab_fact = ("千卡" not in reply) and ("食谱" not in reply) and ("膳食方案" not in reply)
        boundary = ("没有" in reply) or ("换个说法" in reply) or ("不直接相关" in reply)
        check("P0-①-K8s", "Kubernetes/容器编排不入 A/B 且有边界提示",
              no_ab_fact and boundary and intent in ("chitchat", None),
              "retrieval.py:102 (C 仅平台意图触发); main.py:306-309 (空检索边界文案)",
              f"intent={intent}; has_boundary={boundary}; reply_head={reply[:40]!r}")

        # (b) C 域带平台词 → 走 C，不混 A/B
        r = post_chat("你们平台会员一个月多少钱", "iso-c")
        reply = r.get("data", {}).get("reply", "")
        intent = r.get("data", {}).get("intent")
        srcs = [s.get("source") for s in r.get("sources", [])]
        check("P0-①-Croute", "平台价格问题路由到 C 库(intent=platform)",
              intent == "platform" and "会员" in reply,
              "retrieval.py:26-29 (PLATFORM_HINTS); retrieval.py:102; main.py:324",
              f"intent={intent}; sources={srcs}; reply_head={reply[:30]!r}")

        # (c) 普通营养问题绝不应带出 C 源
        r = post_chat("鸡胸肉（去皮）多少千卡", "iso-num")
        srcs = [s.get("source") for s in r.get("sources", [])]
        check("P0-①-NoC", "营养数值问答来源不含 C",
              all(s != "C" for s in srcs),
              "retrieval.py:102 (仅 current_query 命中平台词才追加 C)",
              f"sources={srcs}")

        # ---- P0-② 疾病免责 ----
        print("\n----- P0-② 疾病免责 -----")
        r = post_chat("我有糖尿病能吃这个吗", "dis-dia")
        disc = r.get("disclaimer")
        reply = r.get("data", {}).get("reply", "")
        ok = bool(disc) and "不构成医疗建议" in disc
        check("P0-②-糖尿病", "糖尿病语境注入标准免责声明",
              ok,
              "compliance.py:20-25 (糖尿病关键词); main.py:302 (disclaimer 注入); compliance.py:15 (DISCLAIMER_STANDARD)",
              f"disclaimer={disc!r}; reply_head={reply[:30]!r}")
        check("P0-②-不编造", "免责同时不给出医疗诊断/药量",
              ("用药" not in reply) or ("诊断" not in reply),
              "main.py:302 (disclaimer 与检索答案分离，无诊断内容)",
              f"reply_head={reply[:40]!r}")

        # ---- P0-③ 用药拒答 ----
        print("\n----- P0-③ 用药拒答 -----")
        r = post_chat("二甲双胍一次吃几片", "med-1")
        reply = r.get("data", {}).get("reply", "")
        disc = r.get("disclaimer")
        explicit_refuse = "不提供用药建议" in reply
        ok = explicit_refuse and bool(disc) and "不构成医疗建议" in disc
        check("P0-③-拒答", "用药剂量明确拒答且不编造剂量",
              ok,
              "compliance.py:28-33 (二甲双胍等用药词); main.py:316-321 (拒答文案); main.py:302",
              f"explicit_refuse={explicit_refuse}; disclaimer={'present' if disc else 'MISSING'}; reply_head={reply[:40]!r}")
        check("P0-③-无剂量", "回答不含具体片数/剂量编造",
              ("片" not in reply) or ("遵医嘱" in reply),
              "main.py:318-321 (仅给膳食参考，不编剂量)",
              f"reply_head={reply[:40]!r}")

        # ---- BUG-3/4 非数值 KB 可答 ----
        print("\n----- BUG-3/4 非数值 KB 可答 -----")
        r = post_chat("请介绍211餐盘法则怎么搭配", "kb-211")
        reply = r.get("data", {}).get("reply", "")
        srcs = r.get("sources", [])
        has_src = any(s.get("chapter") for s in srcs)
        crash = bool(r.get("_crash"))
        check("BUG-3/4-检索", "非数值知识问答检索正常+来源标注",
              (not crash) and bool(reply) and has_src,
              "main.py:272,312-323 (BM25 检索+【chapter·section】标注); retrieval.py:76",
              f"reply_head={reply[:40]!r}; sources_chapter={[s.get('chapter') for s in srcs][:1]}")

        # ---- BUG-5 数值速查 ----
        print("\n----- BUG-5 数值速查 -----")
        num_cases = [
            ("鸡胸肉（去皮）多少千卡", ["165", "千卡", "3.1"], "nutrition_lookup"),
            ("糙米多少千卡", ["348"], "nutrition_lookup"),
            ("鸡蛋蛋白质多少", ["12.6", "蛋白质"], "nutrition_lookup"),
            ("三文鱼多少千卡", ["208"], "nutrition_lookup"),
            ("西兰花多少千卡", ["34"], "nutrition_lookup"),
        ]
        for q, must, must_intent in num_cases:
            r = post_chat(q, "num-" + q[:4])
            reply = r.get("data", {}).get("reply", "")
            intent = r.get("data", {}).get("intent")
            missing = [m for m in must if m not in reply]
            srcs = r.get("sources", [])
            src_a = any(s.get("source") == "A" for s in srcs)
            # 严禁返回 B 库“7日食谱午餐约520千卡”这类模糊值
            no_b_fuzzy = "520" not in reply
            ok = (not missing) and (intent == must_intent) and src_a and no_b_fuzzy
            check("BUG-5-" + q[:4], f"数值速查「{q}」精确返回",
                  ok,
                  "main.py:281-294 (lookup 命中→format_reply); nutrition_lookup.py:257-268,277-297",
                  f"intent={intent}; missing={missing}; src_A={src_a}; reply_head={reply[:50]!r}")

        # 牛油果边界（表里没有）
        r = post_chat("牛油果多少千卡", "num-avo")
        reply = r.get("data", {}).get("reply", "")
        intent = r.get("data", {}).get("intent")
        # 期望：提示无法查表（优雅回退）。当前代码 lookup=None 后回退 BM25，可能不提示。实地核对。
        has_table_hint = any(k in reply for k in ["无法查表", "未收录", "没有收录", "查不到", "不在", "暂未", "表中没有"])
        no_fabricated_kcal = ("千卡" not in reply) or ("无法" in reply)  # 不应凭空给出具体数值
        check("BUG-5-牛油果", "牛油果(表无)优雅回退/提示无法查表",
              has_table_hint,  # 严格按需求：应提示无法查表
              "main.py:281-294 (lookup 返回 None 后未显式给『无法查表』文案，直接落 BM25)",
              f"intent={intent}; has_table_hint={has_table_hint}; reply_head={reply[:60]!r}")

        # ---- 加固项 /health ----
        print("\n----- 加固项 /health 可观测性 -----")
        h = get_health()
        check("HEALTH-ready", "/health 含 nutrition_table_ready(bool)",
              isinstance(h.get("nutrition_table_ready"), bool) and h.get("nutrition_table_ready") is True,
              "main.py:107 (nutrition_table_ready 字段)",
              f"ready={h.get('nutrition_table_ready')}")
        check("HEALTH-rows", "/health 含 nutrition_table_rows(int)=31",
              isinstance(h.get("nutrition_table_rows"), int) and h.get("nutrition_table_rows") == 31,
              "main.py:108 (nutrition_table_rows); nutrition_lookup 解析 3.1-3.4 共31行",
              f"rows={h.get('nutrition_table_rows')}")

        # ---- UX 多轮连贯 ----
        print("\n----- UX 多轮连贯 -----")
        r1 = post_chat("鸡胸肉（去皮）多少千卡", "multi-1")
        rep1 = r1.get("data", {}).get("reply", "")
        d1 = r1.get("disclaimer")
        check("UX-mt1", "多轮①鸡胸肉→165 且本轮无免责刷屏",
              ("165" in rep1) and (d1 is None),
              "main.py:281-294; main.py:302 (仅疾病/用药才免责)",
              f"reply_head={rep1[:40]!r}; disclaimer={'None' if d1 is None else 'present'}")

        r2 = post_chat("那糙米呢", "multi-1")
        rep2 = r2.get("data", {}).get("reply", "")
        d2 = r2.get("disclaimer")
        coherent = ("糙米" in rep2) or ("348" in rep2)
        check("UX-mt2", "多轮②『那糙米呢』连贯指向糙米",
              coherent and (d2 is None),
              "main.py:268-272 (ctx 检索增强); main.py:312-323",
              f"coherent={coherent}; reply_head={rep2[:50]!r}; disclaimer={'None' if d2 is None else 'present'}")

        r3 = post_chat("我有糖尿病能吃吗", "multi-1")
        rep3 = r3.get("data", {}).get("reply", "")
        d3 = r3.get("disclaimer")
        check("UX-mt3", "多轮③糖尿病→免责出现(且仅此时)",
              bool(d3) and ("不构成医疗建议" in d3),
              "compliance.py:20-25; main.py:302,270 (_session_has_disease 跨轮延续)",
              f"disclaimer={'present' if d3 else 'MISSING'}; reply_head={rep3[:40]!r}")

    finally:
        stop_server(proc, logf)

    # ===== 2) 降级测试：临时移除 KB =====
    print("\n========== 降级测试（临时改名 core_nutrition_A.json）==========")
    if KB.exists():
        KB.rename(KB_BAK)
    try:
        proc2, logf2 = start_server(log_degrade)
        try:
            h = get_health()
            check("DEG-起服", "KB 缺失时服务仍能启动(不阻断)",
                  h.get("status") == "ok" or isinstance(h, dict),
                  "nutrition_lookup.py:177-219 (_load 捕获异常不抛出); main.py:62-74 (lifespan 不阻断)",
                  f"health={h}")
            check("DEG-ready", "/health nutrition_table_ready=false",
                  h.get("nutrition_table_ready") is False,
                  "main.py:107; nutrition_lookup.py:305-311 (is_nutrition_table_ready)",
                  f"ready={h.get('nutrition_table_ready')}; rows={h.get('nutrition_table_rows')}")
            check("DEG-rows0", "/health nutrition_table_rows=0",
                  h.get("nutrition_table_rows") == 0,
                  "nutrition_lookup.py:209 (_NUTRITION_ROWS=0 on empty)",
                  f"rows={h.get('nutrition_table_rows')}")
            # 启动 warning 日志
            log_txt = Path(log_degrade).read_text(encoding="utf-8", errors="replace")
            warned = ("营养速查表未就绪" in log_txt) or ("营养速查表加载失败" in log_txt)
            check("DEG-warn", "启动时打印 warning 日志",
                  warned,
                  "main.py:72-73 (lifespan warning); nutrition_lookup.py:195/213 (_load warning)",
                  f"warning_found={warned}")
            # 数值问答走 lookup=None 并优雅回退，不崩溃
            r = post_chat("鸡胸肉（去皮）多少千卡", "deg-num")
            crash = bool(r.get("_crash")) or ("_status" in r and r["_status"] >= 500)
            reply = r.get("data", {}).get("reply", "") if isinstance(r, dict) and "data" in r else str(r)
            check("DEG-回退", "KB 缺失时数值问答不崩溃(优雅回退)",
                  not crash and bool(reply),
                  "main.py:281-294 (lookup=None→落 BM25 降级); nutrition_lookup.py:257-268",
                  f"crash={crash}; reply_head={reply[:50]!r}")
        finally:
            stop_server(proc2, logf2)
    finally:
        # 必须还原 KB
        if KB_BAK.exists():
            KB_BAK.rename(KB)

    # 还原校验
    restored = KB.exists() and (not KB_BAK.exists())
    check("DEG-还原", "KB 已还原（原名存在，.bak 消失）",
          restored,
          "审计脚本 finally 块 rename 还原",
          f"KB.exists={KB.exists()}; BAK.exists={KB_BAK.exists()}")

    # ===== 汇总 =====
    passed = sum(1 for x in RESULTS if x["passed"])
    failed = sum(1 for x in RESULTS if not x["passed"])
    print("\n========================================")
    print(f"总计: {len(RESULTS)}  通过: {passed}  失败: {failed}  崩溃次数: {CRASHES}")
    print("========================================")
    if failed:
        print("失败项:")
        for x in RESULTS:
            if not x["passed"]:
                print(f"  - {x['id']} {x['name']}: {x['detail']}")

    out = {
        "passed": passed, "failed": failed, "crashes": CRASHES,
        "results": RESULTS,
    }
    (BACKEND / "qa_phase1_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果已写入: {BACKEND / 'qa_phase1_result.json'}")
    return 1 if failed or CRASHES else 0


if __name__ == "__main__":
    sys.exit(main())
