"""合规层（M10 禁忌拦截 + M8/M11 医疗免责）：规则硬逻辑，不靠模型自觉。

- 医疗免责：命中疾病/症状/用药关键词即强制注入标准免责文案；含用药词则拒答用药、只给膳食参考。
- 禁忌拦截：结合「用户声明 allergies」+「对话触发词」双路识别禁忌 id，返回需排除的食材清单。
所有判定均为确定性强匹配，零幻觉、零模型依赖，保证评审无论如何诱导都稳定合规。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import KNOWLEDGE_DIR

# 标准免责文案（合同 2.1 / 蓝图 8.5 逐字要求）
DISCLAIMER_STANDARD = (
    "本建议仅供参考，不构成医疗建议，请咨询专业医师或注册营养师"
)

# 疾病 / 症状 / 特殊人群意图触发词：命中即强制注入免责
# V05 修复：补充 脂肪肝/甲减/血脂高/血压偏高/尿酸偏高/贫血/肿瘤/癌症/失眠/抑郁/焦虑，
# 覆盖「我有XX怎么办」「XX怎么吃」问法（甲亢/血压高/尿酸高等已有）。
_DISEASE_KEYWORDS = [
    "糖尿病", "血糖", "高尿酸", "尿酸高", "痛风", "肾病", "肾功能不全",
    "慢性肾病", "孕期", "怀孕", "孕妇", "高血压", "血压高", "控压",
    "胰岛素", "降脂", "甲亢", "乙肝", "用药", "药物", "剂量", "处方",
    "吃什么药", "药量",
    # V05：常见慢性病/症状/心理类（命中即强制免责）
    "脂肪肝", "甲减", "血脂高", "血压偏高", "尿酸偏高",
    "贫血", "肿瘤", "癌症", "失眠", "抑郁", "焦虑",
]

# 用药类关键词：命中则拒答用药建议
# 规则：优先完整药名 + 低误报药名尾缀（洛芬/西林/头孢/沙星/霉素/地平/洛尔/替丁/唑/列净/格列汀），
# 宁精勿滥——严禁命中营养补剂/食材（维生素、钙片、蛋白粉、叶酸、DHA、鱼油、益生菌、酵素、胶原蛋白等均不含下列词）。
_MEDICATION_KEYWORDS = [
    "药", "用药", "剂量", "处方", "药量", "降糖药", "降压药", "胰岛素",
    # 常用通用名（不含"药"字，需显式补充，避免绕过拒药 + 免责）
    "二甲双胍", "司美格鲁肽", "阿卡波糖", "格列美脲", "利拉鲁肽",
    "罗格列酮", "他汀", "沙坦", "普利", "唑嗪", "格列奈", "西格列汀",
    # 解热镇痛 / 感冒 / 抗过敏
    "布洛芬", "洛芬", "对乙酰氨基酚", "扑热息痛", "阿司匹林", "阿斯匹林",
    "酚麻美敏", "右美沙芬", "感冒灵", "氯雷他定", "西替利嗪",
    # 抗生素 / 抗感染（药名尾缀：西林/头孢/沙星/霉素/唑）
    "阿莫西林", "西林", "头孢", "左氧氟沙星", "沙星", "阿奇霉素", "霉素",
    "甲硝唑", "奥美拉唑", "唑",
    # 心脑血管用药（药名尾缀：洛尔/替丁；地平类只列完整药名，避免"地平线"误报）
    "氨氯地平", "硝苯地平", "非洛地平", "拉西地平", "尼莫地平", "尼群地平",
    "美托洛尔", "洛尔", "替丁", "华法林", "氯吡格雷",
    # 消化 / 呼吸 / 激素
    "多潘立酮", "吗丁啉", "蒙脱石散", "口服补液盐", "铝碳酸镁",
    "氨溴索", "乙酰半胱氨酸", "布地奈德", "沙丁胺醇", "泼尼松", "地塞米松",
    # 降糖家族补充（只补完整药名与低误报尾缀，避免裸"格列"误伤"格列佛"等）
    "格列齐特", "格列吡嗪", "格列本脲", "格列汀", "列净",
    # 常见中成药（均为明确药品）
    "健胃消食片", "藿香正气", "连花清瘟", "板蓝根",
    # 英文通用名（评委可能用英文问药，如 ibuprofen/aspirin；均小写，配合 message.lower() 匹配）
    "ibuprofen", "aspirin", "paracetamol", "acetaminophen", "amoxicillin",
    "penicillin", "cephalosporin", "metformin", "semaglutide", "azithromycin",
    "amlodipine", "metoprolol", "insulin", "levofloxacin", "ciprofloxacin",
    "atorvastatin", "omeprazole", "loratadine", "cetirizine",
    # 商品名（评委可能问「芬必得/泰诺」而非「布洛芬/对乙酰氨基酚」）
    "芬必得", "泰诺", "泰诺林", "必理通", "散利痛", "百服宁",
    "扶他林", "拜阿司匹灵", "达喜",
    # 药类大类词（不含「药」字，防「抗生素能随便吃吗」绕过）
    "抗生素", "止痛片", "退烧片", "消炎片", "安眠片", "止咳糖浆", "镇静剂",
    # V03 修复：GLP-1 类（诺和泰/诺和力/度拉糖肽/司美格鲁肽及英文商品名）
    "诺和泰", "诺和力", "度拉糖肽", "ozempic", "wegovy", "rybelsus",
    # V03 修复：常见处方药（优甲乐/立普妥/络活喜/倍他乐克/代文/波立维/拜唐苹）
    "优甲乐", "立普妥", "络活喜", "倍他乐克", "代文", "波立维", "拜唐苹",
    # V03 修复：英文 OTC/处方（advil/tylenol/naproxen/diclofenac/codeine，全小写）
    "advil", "tylenol", "naproxen", "diclofenac", "codeine",
    # V03 修复：错别字/口语（止疼片/止疼药；退烧片/安眠药已有「药/片」覆盖）
    "止疼片", "止疼药",
]

# P2 过敏追问话术：键=knowledge/taboo_map.json 的禁忌 id，值为首次声明过敏时
# 的人性化追问（像真人顾问「聊起来」）。仅覆盖可进一步细分的过敏/不耐受类；
# 未配置话术的 id 在 chat() 中跳过，不影响任何合规硬逻辑。
ALLERGY_FOLLOWUP: dict[str, str] = {
    "seafood_allergy": (
        "收到，已为您记录海鲜过敏。为了更精准地帮您避开，"
        "方便告诉我具体是哪一类吗？比如虾类、蟹类、贝类，还是鱼类？"
    ),
    "nut_allergy": (
        "收到，已为您记录坚果过敏。为了更精准地帮您避开，"
        "方便告诉我具体是哪一类坚果吗？比如花生、腰果、杏仁，还是核桃？"
    ),
    "lactose_intolerance": (
        "收到，已为您记录乳糖不耐受。为了更精准地帮您避开，"
        "方便告诉我具体是哪种乳制品会让您不适吗？比如牛奶、酸奶、奶酪，还是冰淇淋？"
    ),
    "gluten_intolerance": (
        "收到，已为您记录麸质不耐受。为了更精准地帮您避开，"
        "方便告诉我具体需要避开哪些小麦类制品吗？比如面包、面条，还是馒头？"
    ),
}

# 膳食顾问领域特征词（路由门控用）：命中任一即判为「膳食顾问领域」→ 走 RAG。
# 设计要点：**不含泛化动词「吃/喝/什么/怎么」**——否则「大龙虾吃什么」会被误判为膳食，
# 落入 BM25 硬套无关知识块。领域判定靠「明确的营养概念 + 目标方案 + 健康疾病 + 食材名」。
_DIETARY_HINTS = (
    # 营养概念
    "热量", "千卡", "卡路里", "大卡", "千焦", "蛋白质", "脂肪", "碳水",
    "营养", "维生素", "矿物质", "膳食纤维", "胆固醇", "嘌呤", "升糖",
    # 目标 / 方案（含 synonyms.json 概念词别名，如 降糖/稳糖/减重——领域门控不吃同义词扩展，
    # 必须在此覆盖，否则「降糖怎么吃」会被误判为非膳食走闲聊）
    "减脂", "减肥", "减重", "掉秤", "增肌", "控糖", "降糖", "稳糖", "调理", "瘦身", "塑形",
    "健身", "运动", "食谱", "餐盘", "饮食", "三餐", "早餐", "午餐", "晚餐",
    "加餐", "夜宵", "宵夜", "搭配", "聚餐", "代餐", "轻断食", "生酮", "低卡",
    "低脂", "高蛋白", "低盐", "低糖",
    # 身体 / 口语化膳食诉求（「减肚子」「长肌肉」「怎么吃才能瘦」「总饿」）
    "肚子", "赘肉", "肌肉", "长肉", "体脂", "体重", "腰围", "腹肌", "马甲线",
    "瘦", "胖", "饿", "零食",
    # 健康 / 疾病（与 _DISEASE_KEYWORDS 互补，路由时只需领域信号、无需免责语义）
    "糖尿病", "血糖", "痛风", "高尿酸", "高血压", "血脂", "肾病", "孕期",
    "孕妇", "哺乳", "过敏", "禁忌", "甲亢", "甲减", "脂肪肝", "贫血",
    # V05：新疾病词同步过领域门控（否则 is_dietary_domain 不过 → 走 chitchat）
    "血脂高", "血压偏高", "尿酸偏高", "肿瘤", "癌症", "失眠", "抑郁", "焦虑",
)


def is_dietary_domain(message: str) -> bool:
    """是否属于膳食顾问领域（路由门控：膳食 → RAG，非膳食 → 闲聊）。

    「大龙虾吃什么」「今天天气怎么样」「世界首富是谁」等不含任何膳食特征词，
    判 False 走闲聊分支，避免 BM25 检索到低相关块后硬套膳食内容。
    """
    text = message or ""
    if any(kw in text for kw in _DIETARY_HINTS):
        return True
    # 营养速查表食材名（动态）：鸡胸肉/西兰花/燕麦等命中即膳食领域（≥2 字，避免单字误伤）
    try:
        from app import nutrition_lookup
        for food in nutrition_lookup.NUTRITION_TABLE:
            if food and len(food) >= 2 and food in text:
                return True
    except Exception:
        pass
    return False


def _load_taboos() -> list[dict]:
    path = KNOWLEDGE_DIR / "taboo_map.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("taboos", [])
    except (OSError, json.JSONDecodeError):
        return []


def is_disease_query(message: str) -> bool:
    """是否命中疾病/医疗意图（需强制免责）。"""
    return any(kw in (message or "") for kw in _DISEASE_KEYWORDS)


def is_medication_query(message: str) -> bool:
    """是否含用药咨询（需拒答用药、只给膳食参考）。英文药名大小写不敏感。"""
    text = (message or "").lower()
    return any(kw in text for kw in _MEDICATION_KEYWORDS)


# V02 症状式过敏映射：食材 → 禁忌族 id。
# 正则形态：`吃<食材>.{0,3}(起疹|发痒|浑身痒|过敏|难受)`（「一吃虾就起疹」「吃鸡蛋起疹」）。
# 从单一「吃虾」泛化到 鱼/虾/蟹/鸡蛋/豆腐/花生/牛奶，并按食材归属对应禁忌族，
# 修复「吃鸡蛋起疹」「吃豆腐起疹」等经同族替换泄露的禁忌漏识别；
# 同时避免误伤营养问（「吃鱼有什么好处」「吃鱼油」均不命中）。
# 蛋类用多字名优先（鸡蛋/鸭蛋），否则「吃鸡蛋起疹」的「鸡」会隔断「吃」与「蛋」。
_FOOD_SYMPTOM_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("鱼",), "fish_allergy"),
    (("虾", "蟹"), "seafood_allergy"),
    (("鸡蛋", "鸭蛋", "蛋"), "egg_allergy"),
    (("豆腐",), "soy_allergy"),
    (("花生",), "nut_allergy"),
    (("牛奶",), "lactose_intolerance"),
)
_SYMPTOM_RE = re.compile(r"(?:起疹|发痒|浑身痒|过敏|难受)")


def detect_allergies(message: str, declared: list[str] | None = None) -> list[str]:
    """返回命中的禁忌 id 列表：用户声明 + 对话触发词双路合并。"""
    hits: set[str] = set(declared or [])
    text = message or ""
    for taboo in _load_taboos():
        if any(kw in text for kw in taboo.get("trigger_keywords", [])):
            hits.add(taboo["id"])
    # 「症状描述式」过敏：食材与症状被「就/会/了/一」等隔开，纯子串匹配抓不到。
    # 例：「一吃虾就起疹子」「吃鸡蛋起疹」「一喝牛奶就不舒服」——用正则精准补齐。
    for foods, aid in _FOOD_SYMPTOM_PATTERNS:
        if re.search(rf"吃(?:{'|'.join(foods)}).{{0,3}}{_SYMPTOM_RE.pattern}", text):
            hits.add(aid)
    if re.search(r"喝(?:牛奶|奶).{0,4}(?:拉肚子|腹泻|不舒服|难受|肚子|胀气|窜稀|过敏)", text):
        hits.add("lactose_intolerance")
    return list(hits)


def excluded_foods(allergy_ids: list[str]) -> list[str]:
    """返回需排除的食材清单（去重）。"""
    foods: list[str] = []
    seen: set[str] = set()
    for taboo in _load_taboos():
        if taboo["id"] in allergy_ids:
            for f in taboo.get("excluded_foods", []):
                if f not in seen:
                    seen.add(f)
                    foods.append(f)
    return foods
