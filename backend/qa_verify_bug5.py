"""Independent QA verification of BUG-5 fix — hits the LIVE server on port 8200.

Does NOT import app.main / does NOT call the implementer's self-test.
Asserts the actual HTTP JSON contract: data.reply / data.intent / disclaimer.
"""
from __future__ import annotations

import json
import sys

import requests

BASE = "http://127.0.0.1:8200"
ENDPOINT = f"{BASE}/api/chat"

# (label, query, session_id, expectations)
# expectations: dict with optional keys:
#   contains_all: list[str]  -> reply must contain every substring
#   intent: str              -> data.intent must equal
#   not_intent: str          -> data.intent must NOT equal (e.g. not nutrition_lookup)
#   disclaimer: bool         -> disclaimer must be present (non-null)
#   disclaimer_absent: bool  -> disclaimer must be null
#   reject_med: bool         -> reply must mention medication refusal (不提供用药建议 / 遵医嘱)
CASES = [
    ("numeric_鸡胸肉", "鸡胸肉多少千卡", "qa-1",
     {"contains_all": ["165", "千卡", "3.1 肉类与海鲜"], "intent": "nutrition_lookup"}),
    ("numeric_糙米", "糙米多少千卡", "qa-2",
     {"contains_all": ["348", "千卡"], "intent": "nutrition_lookup"}),
    ("numeric_鸡蛋蛋白", "鸡蛋多少蛋白质", "qa-3",
     {"contains_all": ["12.6", "蛋白质"], "intent": "nutrition_lookup"}),
    ("numeric_三文鱼", "三文鱼多少千卡", "qa-4",
     {"contains_all": ["208", "千卡"], "intent": "nutrition_lookup"}),
    ("numeric_西兰花", "西兰花多少千卡", "qa-5",
     {"contains_all": ["34", "千卡"], "intent": "nutrition_lookup"}),
    # NON-numeric must NOT be hijacked by nutrition_lookup
    ("nonnumeric_减脂早餐", "减脂早餐吃什么", "qa-6",
     {"intent": "nutrition_qa", "not_intent": "nutrition_lookup",
      "contains_all": ["【"], "disclaimer_absent": True}),
    # DISEASE -> disclaimer present
    ("disease_糖尿病", "我有糖尿病吃什么", "qa-7",
     {"disclaimer": True, "contains_any_disclaimer": ["免责", "仅供参考", "不构成医疗"]}),
    # MEDICATION -> reject + disclaimer
    ("medication_司美格鲁肽", "司美格鲁肽减肥怎么吃", "qa-8",
     {"reject_med": True, "disclaimer": True,
      "contains_any_disclaimer": ["免责", "仅供参考", "不构成医疗"]}),
    # PLATFORM -> source C, no crash
    ("platform_会员", "会员多少钱", "qa-9",
     {"intent": "platform",
      "contains_any_platform": ["会员", "套餐", "价格", "订阅", "平台"], "disclaimer_absent": True}),
    # EDGE: food NOT in table -> graceful fallback (no error, no nutrition_lookup)
    ("edge_牛油果_notintable", "牛油果多少千卡", "qa-10",
     {"not_intent": "nutrition_lookup", "not_crash": True}),
]


def run():
    results = []
    server_errors = []
    import time
    run_stamp = int(time.time() * 1000)
    for idx, (label, q, sid, exp) in enumerate(CASES):
        # Use a unique session per case (per run) so the cross-turn disease
        # disclaimer carryover (_session_has_disease) can't contaminate a later
        # non-disease case that reuses the same session id.
        sid = f"qa-{run_stamp}-{idx}"
        body = {"message": q, "user_id": "qa", "session_id": sid}
        try:
            resp = requests.post(ENDPOINT, json=body, timeout=30)
        except requests.RequestException as e:
            results.append((label, "FAIL", f"request error: {e}", None))
            server_errors.append((label, str(e)))
            continue

        if resp.status_code >= 500:
            server_errors.append((label, f"HTTP {resp.status_code}: {resp.text[:300]}"))
            results.append((label, "FAIL", f"HTTP {resp.status_code} server error", resp.text[:300]))
            continue

        try:
            payload = resp.json()
        except ValueError:
            results.append((label, "FAIL", f"non-JSON body (HTTP {resp.status_code})", resp.text[:300]))
            continue

        data = payload.get("data", {}) or {}
        reply = data.get("reply", "") or ""
        intent = data.get("intent", "")
        # NOTE: disclaimer is a TOP-LEVEL field of UnifiedResponse, a sibling of
        # `data` (not nested inside it). Reading it from `data` would be wrong.
        disclaimer = payload.get("disclaimer")
        reasons = []

        # intent checks
        if "intent" in exp and intent != exp["intent"]:
            reasons.append(f"intent={intent!r} expected {exp['intent']!r}")
        if "not_intent" in exp and intent == exp["not_intent"]:
            reasons.append(f"intent unexpectedly == {intent!r}")

        # reply substring checks
        if "contains_all" in exp:
            for s in exp["contains_all"]:
                if s not in reply:
                    reasons.append(f"reply missing {s!r}")
        if "contains_any_disclaimer" in exp:
            if not any(s in (disclaimer or "") for s in exp["contains_any_disclaimer"]):
                reasons.append(f"disclaimer {disclaimer!r} lacks 免责/仅供参考/不构成医疗")
        if "contains_any_platform" in exp:
            # platform reply should come from C lib (about membership/price). We just check it's non-empty
            # and not a nutrition_lookup error; rely on intent/platform source.
            if not any(s in reply for s in exp["contains_any_platform"]):
                reasons.append(f"platform reply lacks membership/price keywords: {reply[:80]!r}")
        if exp.get("reject_med"):
            if not ("不提供用药建议" in reply or "遵医嘱" in reply):
                reasons.append(f"med reply lacks 不提供用药建议/遵医嘱: {reply[:100]!r}")

        # disclaimer presence checks
        if exp.get("disclaimer") and disclaimer in (None, ""):
            reasons.append("disclaimer expected but null")
        if exp.get("disclaimer_absent") and disclaimer not in (None, ""):
            reasons.append(f"disclaimer unexpectedly present: {disclaimer!r}")

        status = "PASS" if not reasons else "FAIL"
        if status == "PASS":
            detail = f"intent={intent!r} reply_head={reply[:60]!r}"
        else:
            detail = "; ".join(reasons) + f" | intent={intent!r} reply={reply[:120]!r} disclaimer={disclaimer!r}"
        results.append((label, status, detail, reply))

    # also verify /health
    try:
        h = requests.get(f"{BASE}/health", timeout=10)
        health_ok = h.status_code == 200 and h.json().get("status") == "ok"
    except requests.RequestException:
        health_ok = False

    return results, server_errors, health_ok


if __name__ == "__main__":
    results, server_errors, health_ok = run()
    print("=" * 70)
    print("HEALTH endpoint ok:", health_ok)
    print("=" * 70)
    passed = failed = 0
    for label, status, detail, _ in results:
        print(f"[{status}] {label}: {detail}")
        if status == "PASS":
            passed += 1
        else:
            failed += 1
    print("=" * 70)
    print(f"TOTAL={len(results)} PASS={passed} FAIL={failed}  server_5xx={len(server_errors)}")
    if server_errors:
        print("SERVER 5xx:")
        for lbl, err in server_errors:
            print(f"  {lbl}: {err}")
    # summary table for the report
    print("\n# SUMMARY_TABLE")
    for label, status, _, _ in results:
        print(f"{label}\t{status}")
    sys.exit(0 if failed == 0 and not server_errors else 1)
