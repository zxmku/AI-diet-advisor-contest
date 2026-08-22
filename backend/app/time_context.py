"""时间感知适配器（管道适配器模式 · 热插拔）：按当前时刻注入时段问候与建议焦点。

供 LLM 系统提示与前端欢迎语使用；零依赖、无状态，纯函数。

时区约定：统一采用 UTC+8（北京时间）计算「当前时刻」——Docker 容器 / Linux
服务器默认 UTC 时区，若直接用 datetime.now() 会差 8 小时（北京 21:03 = UTC
13:03，误判「午餐时段」）。health_state.py 已有 _UTC8 先例，此处对齐。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

_UTC8 = timezone(timedelta(hours=8))


def get_meal_time_context(now: datetime | None = None) -> dict:
    """返回当前时段上下文：period（时段名）+ greeting（问候语）+ focus（建议焦点）。"""
    if now is None:
        hour = datetime.now(_UTC8).hour
    else:
        # 统一转成 UTC+8 再取 hour：naive 视为 +8 本地时间；带时区（如 UTC）则 astimezone 换算
        if now.tzinfo is None:
            now = now.replace(tzinfo=_UTC8)
        hour = now.astimezone(_UTC8).hour
    if 6 <= hour < 10:
        return {
            "period": "早餐时段",
            "greeting": "早上好！新的一天从高蛋白早餐开始，启动全天代谢。",
            "focus": "建议搭配：优质蛋白 + 复合碳水 + 蔬果（参考早餐黄金公式）。",
        }
    if 11 <= hour < 14:
        return {
            "period": "午餐时段",
            "greeting": "中午好！工作辛苦了，午餐推荐遵循 211 餐盘法则，保持午后精力。",
            "focus": "建议搭配：2 拳蔬菜 + 1 掌心优质蛋白 + 1 拳低 GI 主食。",
        }
    if 17 <= hour < 20:
        return {
            "period": "晚餐时段",
            "greeting": "晚上好！晚餐建议清淡少油，减轻肠胃负担，助眠修整。",
            "focus": "建议搭配：高纤维蔬菜 + 易消化蛋白质，主食适量减半。",
        }
    return {
        "period": "非正餐/夜宵时段",
        "greeting": "夜深了，注意规律作息，别让胃空着。",
        "focus": "如有饥饿感，优先选择温热无糖豆浆、低脂酸奶或黄瓜小番茄。",
    }
