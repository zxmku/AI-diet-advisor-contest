# -*- coding: utf-8 -*-
"""HealthPick 提交包固化打包脚本（一键出包 + 自动校验）。

用法：
    python 打包脚本_pack_release.py            # 全流程：pytest → 出包 → 敏感扫描 → 写标注
    python 打包脚本_pack_release.py --skip-tests  # 跳过 pytest（自动化任务已跑过时用）

输出：大赛根/40_提交包/（覆盖重建），含：
    10_项目源码_healthpick/  （git archive HEAD 解包，已删 design/）
    50_验证报告/              （拷贝大赛根最新验证报告）
    README.md                 （标注 HEAD 与打包时间）
    提交前清洗清单.md
扫描红线：*.env(除 .example) / *.vsec / local.key / *.db* / *.log / *.pyc /
          __pycache__ / backend/data / .git / 内部词（团长/成员名/本机路径）→ 必须 0 命中。

设计意图：把「打包」变成可复现的一次性动作，提交包永远与 HEAD 同步，
不再存在「旧版提交包」——明早自动化任务直接调本脚本出冻结版。
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent       # 10_项目源码_healthpick/
BASE = SCRIPT_DIR.parent                            # 大赛根
PKG = BASE / "40_提交包"
REPORT_DIR = BASE / "50_验证报告"
PY = r"C:/Users/24771/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

# 内部称谓/本机路径（清洗红线）；sk- 用真实 Key 格式（长度≥16），避免误伤
# 测试假 Key（sk-test-fake）与脱敏正则（secret_redaction.py 的 sk- 字面量）
BANNED = ("团长", "严过关", "寇豆码", "许清楚", "高见远", "齐活林", "子星明",
          "C:/Users", "24771", "197609")
KEY_PAT = re.compile(r"sk-[A-Za-z0-9]{16,}")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--force-replace", action="store_true",
                    help="直接重建 40_提交包（需删除权限；普通沙箱请构建到 40_提交包_new）")
    args = ap.parse_args()

    head = run(["git", "-C", str(SCRIPT_DIR), "log", "--oneline", "-1"]).stdout.strip()
    print(f"[1/6] HEAD = {head}")

    if not args.skip_tests:
        print("[2/6] pytest 全量校验...")
        p = run([PY, "-m", "pytest", str(SCRIPT_DIR / "tests"), "-q"])
        last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "(空)"
        print(f"      {last}")
        if p.returncode != 0 or "passed" not in last or "failed" in last:
            print("      ❌ 测试未全绿，拒绝打包")
            return 1
    else:
        print("[2/6] 跳过 pytest（--skip-tests）")

    # ── 3/6 出包：git archive → 临时目录 → 删 design → 挪入提交包 ──
    print("[3/6] git archive HEAD 出包...")
    tmp = Path(tempfile.mkdtemp(prefix="hp_pkg_"))  # 系统 TEMP，删除失败也不污染大赛根
    tar = tmp / "pkg.tar"
    run(["git", "-C", str(SCRIPT_DIR), "archive", "--format=tar", "HEAD", "-o", str(tar)])
    if not tar.exists():
        print("      ❌ git archive 失败")
        return 1
    # tar 解包到 src 目录
    src_dir = tmp / "src"
    src_dir.mkdir(exist_ok=True)
    run(["tar", "-xf", str(tar), "-C", str(src_dir)])
    try:
        tar.unlink(missing_ok=True)
    except OSError:
        pass  # 沙箱删除受限时忽略（临时目录残留无碍）

    # 删 design/（旧设计样板，含内部注记，不进包）
    design = src_dir / "design"
    if design.exists():
        try:
            shutil.rmtree(design)
        except OSError:
            pass  # git archive HEAD 本身不含 design/，此处仅保险
        print("      （已删 design/）")

    # ── 4/6 构建提交包 ──
    # 目标目录：--force-replace 时直接重建 40_提交包（需删除权限/提权）；
    # 否则构建到 40_提交包_new/（不碰旧包，切换由提权命令完成）。
    target = PKG if args.force_replace else PKG.parent / "40_提交包_new"
    if target.exists():
        try:
            shutil.rmtree(target)
        except OSError:
            print(f"      ⚠️ 无法删除 {target.name}/（沙箱删除受限）")
            if args.force_replace:
                print("      → 请用提权方式运行本脚本（dangerouslyDisableSandbox）")
                return 1
    print(f"[4/6] 构建 {target.name}/...")
    shutil.copytree(src_dir, target / "10_项目源码_healthpick")
    # 验证报告拷贝（取最新）
    reports = sorted(REPORT_DIR.glob("*.md")) if REPORT_DIR.exists() else []
    if reports:
        latest = reports[-1]
        (target / "50_验证报告").mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest, target / "50_验证报告" / latest.name)
        print(f"      已拷验证报告: {latest.name}")
    try:
        shutil.rmtree(tmp, ignore_errors=True)
    except OSError:
        pass

    # ── 5/6 敏感扫描（0 命中才合格）──
    print("[5/6] 敏感扫描...")
    issues: list[str] = []
    for f in target.rglob("*"):
        if f.is_file() and f.name == "README.md" and f.parent == target:
            continue  # 包根 README 由下方重写
        rel = f.relative_to(target).as_posix()
        if f.is_file():
            low = rel.lower()
            if (low.endswith(".env") and ".example" not in low) or low.endswith(".vsec") \
               or f.name == "local.key" or low.endswith(".db") or low.endswith(".db-wal") \
               or low.endswith(".db-shm") or low.endswith(".log") or low.endswith(".pyc") \
               or "/__pycache__/" in low or "/backend/data/" in low or "/.git/" in low:
                issues.append(f"敏感文件: {rel}")
            if low.endswith((".py", ".md", ".html", ".json", ".txt", ".yml", ".yaml", ".ps1", ".bat")):
                try:
                    txt = f.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for b in BANNED:
                    if b in txt:
                        issues.append(f"内部词[{b}]: {rel}")
                if KEY_PAT.search(txt):
                    issues.append(f"疑似真实Key: {rel}")
    if issues:
        print("      ❌ 扫描命中：")
        for i in issues[:15]:
            print(f"        - {i}")
        return 1
    print("      敏感文件 0 / 内部词 0 ✓")

    # ── 6/6 写包根 README 标注 ──
    print("[6/6] 写 README 标注...")
    n_files = sum(1 for _ in target.rglob("*") if _.is_file())
    size_kb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) // 1024
    (target / "README.md").write_text(
        "## HealthPick 提交包（自动生成）\n\n"
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}\n"
        f"- HEAD：`{head.split()[0]}`（{head.split(maxsplit=1)[1] if len(head.split())>1 else ''}）\n"
        f"- 内容：{n_files} 文件 / {size_kb} KB\n"
        "- 敏感扫描：0 命中（.env/.db/日志/内部词全排除）\n"
        "- 验证报告：见 50_验证报告/\n",
        encoding="utf-8",
    )
    print(f"      {n_files} 文件 / {size_kb} KB")
    print(f"\n✅ 打包完成：{target}")
    if target.name == "40_提交包_new":
        print("   ⚠️ 旧包未替换：请用提权运行 --force-replace，或手动把 _new 切换为 40_提交包")
    print("   入口：双击包内 10_项目源码_healthpick/一键启动.bat（或 Docker compose）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
