"""素材 → 结构化 JSON 加工脚本（MOD-01，可重复跑）。

从只读素材目录（04_mdagent_markdown/）解析素材 A/B/C 的 markdown，
按「章 / 节」切分为检索块，产出：
- core_nutrition_A.json  素材 A：营养素基础/211餐盘/时令食材/速查表/特殊人群/禁忌
- core_plans_B.json      素材 B：三套方案+7日循环食谱+营养素目标+替换指南
- platform_C.json        素材 C：平台介绍/会员定价/企业SaaS/API/案例（物理隔离独立文件）
- taboo_map.json         禁忌映射表（蓝图 5.5 七类全量收录，结构见接口契约 2.4）
- synonyms.json          同义词表（降糖=控糖=稳糖、减重=减脂、健身=增肌、尿酸高=痛风 等）

铁律：
- 数值零容错：分块正文逐字取自素材原文，脚本不做任何改写/推算；
- chapter 字段与素材原文标题逐字一致（溯源准确，错 = 准确性事故）；
- 仅剔除转换噪点行（生成工具水印行、跨页表格续表标记），不计入内容；
- 素材 C 的会员价格/企业方案只出现在 platform_C.json，绝不混入 A/B。

用法：
    python scripts/build_knowledge.py
    # 素材目录可用环境变量 HEALTHPICK_MATERIALS_DIR 覆盖
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = KNOWLEDGE_DIR.parent            # 10_项目源码_healthpick/
CONTEST_ROOT = PROJECT_ROOT.parent             # 大赛根目录
MATERIALS_DIR = Path(
    os.environ.get("HEALTHPICK_MATERIALS_DIR", str(CONTEST_ROOT / "04_mdagent_markdown"))
)

VERSION = "1.0.0"

# 章标题：行首「第X章」（素材原文标题，逐字保留）
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十]+章")
# 节标题：行首「数字.数字」（如 1.1 / 2.3 / 3.10），排除「1. 列表项」
SECTION_RE = re.compile(r"^\d+\.\d+\s*\S")

# 转换噪点行（非内容：生成工具水印、跨页续表标记），剔除并在此声明
NOISE_LINES = {"Kimi 生成", "Table 3 – continued", "京/上海/广州/深圳）"}

SOURCES = [
    {
        "source": "A",
        "file": "素材A_2026秋季健康膳食指南.md",
        "title": "2026 秋季健康膳食指南",
        "output": "core_nutrition_A.json",
    },
    {
        "source": "B",
        "file": "素材B_个性化饮食计划方案.md",
        "title": "个性化饮食计划方案",
        "output": "core_plans_B.json",
    },
    {
        "source": "C",
        "file": "素材C_健康优选平台服务白皮书.md",
        "title": "健康优选平台服务白皮书",
        "output": "platform_C.json",
    },
]

INTRO_SECTION = "本章概述"  # 章内、首个节标题之前的导语内容的 section 名


def parse_markdown(source_id: str, md_path: Path) -> list[dict]:
    """按章/节切分 markdown 为检索块列表（契约 2.2 四字段）。

    章标题行、节标题行逐字保留为 chapter/section；正文行逐字拼接，
    不做改写。文档首个章标题之前的封面行（标题/版本）不入库。
    """
    chunks: list[dict] = []
    chapter: str | None = None
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if chapter is None or not buffer:
            buffer.clear()
            return
        chunks.append(
            {
                "source": source_id,
                "chapter": chapter,
                "section": section or INTRO_SECTION,
                "content": "\n".join(buffer),
            }
        )
        buffer.clear()

    for raw_line in md_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.strip() in NOISE_LINES:
            continue
        if CHAPTER_RE.match(line):
            flush()
            chapter = line.strip()
            section = None
            continue
        if chapter is None:
            continue  # 封面行（标题/版本信息）不入库
        if SECTION_RE.match(line):
            flush()
            section = line.strip()
            continue
        buffer.append(line)
    flush()
    return chunks


def build_library(spec: dict) -> dict:
    """解析单个素材并返回库 JSON 对象。"""
    md_path = MATERIALS_DIR / spec["file"]
    if not md_path.is_file():
        raise FileNotFoundError(f"素材文件缺失: {md_path}")
    chunks = parse_markdown(spec["source"], md_path)
    return {
        "version": VERSION,
        "source": spec["source"],
        "title": spec["title"],
        "source_file": spec["file"],
        "chunks": chunks,
    }


# ── 禁忌映射表（蓝图 5.5 七类全量收录，结构见接口契约 2.4）──
# disclaimer_type: allergy=禁忌/过敏模板 | disease=疾病/症状标准免责模板（蓝图 8.5）
TABOO_MAP = {
    "version": VERSION,
    "taboos": [
        {
            "id": "seafood_allergy",
            "name": "海鲜过敏",
            "trigger_keywords": ["海鲜过敏", "虾过敏", "吃虾起疹", "海产品过敏"],
            "excluded_foods": ["虾仁", "三文鱼", "鳕鱼", "鲈鱼", "龙利鱼", "蛤蜊", "鱿鱼"],
            "limits": None,
            "disclaimer_type": "allergy",
        },
        {
            "id": "gout",
            "name": "痛风/高尿酸",
            "trigger_keywords": ["痛风", "尿酸高", "高尿酸"],
            "excluded_foods": ["动物内脏", "浓肉汤", "沙丁鱼", "啤酒"],
            "limits": "严格限制嘌呤",
            "disclaimer_type": "allergy",
        },
        {
            "id": "kidney_disease",
            "name": "肾病",
            "trigger_keywords": ["肾病", "肾功能不全", "慢性肾病"],
            "excluded_foods": ["香蕉", "土豆", "菠菜", "牛油果"],
            "limits": "蛋白质 0.6-0.8 克/公斤体重，限盐限钾",
            "disclaimer_type": "disease",
        },
        {
            "id": "gluten_intolerance",
            "name": "Gluten 不耐受（乳糜泻）",
            "trigger_keywords": ["乳糜泻", "麸质不耐受", "Gluten", "gluten"],
            "excluded_foods": ["全麦面包", "全麦馒头", "小麦制品", "大麦制品", "黑麦制品"],
            "limits": "严格避免小麦、大麦、黑麦及其制品",
            "disclaimer_type": "allergy",
        },
        {
            "id": "pregnancy",
            "name": "孕期",
            "trigger_keywords": ["孕期", "怀孕", "孕妇"],
            "excluded_foods": ["生鱼片", "溏心蛋"],
            "limits": "避免生食，限制咖啡因<200 毫克/日，禁酒",
            "disclaimer_type": "disease",
        },
        {
            "id": "lactation",
            "name": "哺乳期",
            "trigger_keywords": ["哺乳期", "母乳喂养", "喂奶"],
            "excluded_foods": ["鲨鱼", "剑鱼", "方头鱼"],
            "limits": "避免高汞鱼类，注意过敏原传递",
            "disclaimer_type": "disease",
        },
        {
            "id": "hypertension",
            "name": "控压（高血压）",
            "trigger_keywords": ["高血压", "血压高", "控压"],
            "excluded_foods": ["酱油", "腌制品", "加工肉类", "浓汤宝", "方便面调料包"],
            "limits": "每日食盐<5 克，警惕隐形盐",
            "disclaimer_type": "disease",
        },
    ],
}

# ── 同义词表（意图识别与检索扩展用）──
SYNONYMS = {
    "version": VERSION,
    "synonyms": [
        {"canonical": "控糖", "aliases": ["控糖", "降糖", "稳糖"]},
        {"canonical": "减脂", "aliases": ["减脂", "减重", "减肥"]},
        {"canonical": "增肌", "aliases": ["增肌", "健身"]},
        {"canonical": "痛风", "aliases": ["痛风", "尿酸高", "高尿酸"]},
        {"canonical": "热量", "aliases": ["热量", "卡路里", "千卡", "大卡"]},
        {"canonical": "高血压", "aliases": ["高血压", "控压", "血压高"]},
        {"canonical": "血糖高", "aliases": ["血糖高", "血糖偏高", "糖尿病前期"]},
    ],
}

# ── 数值抽检清单（≥20 处，逐字子串断言；任一缺失即构建失败）──
VERIFY_SNIPPETS: list[tuple[str, str]] = [
    # 素材 A：速查表关键数值
    ("A", "| 鸡胸肉（去皮） | 165 | 31.0 | 3.6 | 85 | 减脂、增肌 |"),
    ("A", "| 牛里脊 | 125 | 22.0 | 4.0 | 65 | 增肌、补铁 |"),
    ("A", "| 三文鱼 | 208 | 20.4 | 13.4 | 55 | 增肌、健脑 |"),
    ("A", "| 鳕鱼 | 82 | 17.8 | 0.7 | 43 | 减脂、儿童 |"),
    ("A", "| 虾仁 | 93 | 18.6 | 1.0 | 193 | 减脂、增肌 |"),
    ("A", "| 鸡蛋（全蛋） | 155 | 12.6 | 11.1 | 372 | 一般人群 |"),
    ("A", "| 北豆腐 | 98 | 12.2 | 4.8 | 138 | 减脂、素食 |"),
    ("A", "| 希腊酸奶（无糖） | 97 | 10.0 | 5.0 | 110 | 减脂、增肌 |"),
    ("A", "| 奶酪（切达） | 403 | 25.0 | 33.0 | 721 | 增肌、儿童 |"),
    ("A", "| 糙米 | 348 | 76.0 | 3.4 | 50 | 减脂、控糖 |"),
    ("A", "| 燕麦 | 338 | 61.0 | 10.0 | 55 | 减脂、控糖 |"),
    ("A", "| 白米饭 | 130 | 28.0 | 0.4 | 73 | 增肌 |"),
    ("A", "| 红薯 | 86 | 20.0 | 3.0 | 54 | 减脂、一般 |"),
    ("A", "| 全麦面包 | 247 | 41.0 | 7.0 | 50 | 减脂、控糖 |"),
    ("A", "| 藜麦 | 368 | 64.0 | 7.0 | 53 | 减脂、素食 |"),
    ("A", "| 西兰花 | 34 | 4.3 | 2.6 | 89 | 白灼、蒜蓉炒 |"),
    ("A", "| 菠菜 | 23 | 3.6 | 2.2 | 32 | 蒜蓉炒、做汤 |"),
    ("A", "每日添加糖不超过25 克"),
    ("A", "成年人每日推荐摄入量为体重（公斤）×1.0-1.2 克"),
    ("A", "膳食纤维每日推荐摄入25-30 克"),
    # 素材 B：营养素目标与食谱克数
    ("B", "| 总热量 | 1200-1500 千卡 | 1500-1800 千卡 | 根据基础代谢个体化调整 |"),
    ("B", "| 蛋白质 | 60-75 克 | 75-90 克 | 优先鸡胸肉、鱼、豆腐 |"),
    ("B", "| 总热量 | 2500-3000 千卡 | 2200-2600 千卡 | 根据体重和训练强度调整 |"),
    ("B", "| 蛋白质 | 130-160 克 | 130-160 克 | 保持稳定高摄入 |"),
    ("B", "燕麦片40 克+ 牛奶200 毫升+ 蓝莓50 克+ 核桃10 克"),
    ("B", "空腹血糖6.1-7.0 mmol/L（糖尿病前期）"),
    # 素材 C：价格（只许出现在 C 库）
    ("C", "价格：59 元/月或599 元/年（省109 元）"),
    ("C", "价格：199 元/月或1999 元/年（省389 元）"),
    ("C", "| 100-300 人 | 5.8 万元/年 | 基础 SaaS+4 场讲座 + 体检方案 |"),
    ("C", "| 初创版 | 100,000 次 | 999 元/月 |"),
]

# 素材 C 价格特征串：绝不允许出现在 A/B 库（红线：C 不混入营养库）
_C_PRICE_MARKERS = ["元/月", "元/年", "万元/年", "元/次", "元/场", "元/周", "元/份"]


def write_json(name: str, payload: dict) -> Path:
    """写 JSON（UTF-8、不转义中文、两空格缩进）。"""
    path = KNOWLEDGE_DIR / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def verify_libraries(libs: dict[str, dict]) -> list[str]:
    """构建后校验：数值抽检 + C 库隔离红线。返回错误列表（空 = 通过）。"""
    errors: list[str] = []
    for source_id, snippet in VERIFY_SNIPPETS:
        haystack = json.dumps(libs[source_id]["chunks"], ensure_ascii=False)
        if snippet not in haystack:
            errors.append(f"数值抽检失败[{source_id}]: {snippet!r} 未在分块中找到")
    for source_id in ("A", "B"):
        haystack = json.dumps(libs[source_id]["chunks"], ensure_ascii=False)
        for marker in _C_PRICE_MARKERS:
            if marker in haystack:
                errors.append(f"隔离红线违规: 素材 C 价格标记 {marker!r} 混入 {source_id} 库")
    return errors


def main() -> int:
    """加工全部素材并落盘，跑数值抽检与隔离校验。"""
    libs: dict[str, dict] = {}
    for spec in SOURCES:
        lib = build_library(spec)
        libs[spec["source"]] = lib
        out = write_json(spec["output"], lib)
        print(f"[ok] {spec['output']}: {len(lib['chunks'])} 块 -> {out}")

    write_json("taboo_map.json", TABOO_MAP)
    print(f"[ok] taboo_map.json: {len(TABOO_MAP['taboos'])} 类禁忌")
    write_json("synonyms.json", SYNONYMS)
    print(f"[ok] synonyms.json: {len(SYNONYMS['synonyms'])} 组同义词")

    errors = verify_libraries(libs)
    if errors:
        for err in errors:
            print(f"[FAIL] {err}", file=sys.stderr)
        return 1
    print(f"[ok] 数值抽检 {len(VERIFY_SNIPPETS)} 处全部命中；C 库隔离校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
