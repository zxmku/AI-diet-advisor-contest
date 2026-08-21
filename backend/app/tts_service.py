"""语音输出适配器（管道适配器模式 · 热插拔）：文本 → 拟真人神经语音（edge-tts，微软）。

- 零 API Key、零成本；依赖缺失时优雅降级（tts_available()=False → 路由 503 → 前端隐藏喇叭）。
- 独立节点，不触碰业务核心（检索/禁忌/免责均不动）。
"""
from __future__ import annotations

import io
import logging

logger = logging.getLogger("healthpick.tts")

VOICE_DEFAULT = "zh-CN-XiaoxiaoNeural"  # 温暖知性女声；可切 zh-CN-YunxiNeural（阳光男声）

try:
    import edge_tts  # type: ignore

    _EDGE_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    _EDGE_AVAILABLE = False
    logger.warning("edge-tts 未安装，语音输出降级不可用：%s", exc)


def tts_available() -> bool:
    """语音合成是否可用（edge-tts 已安装且可导入）。"""
    return _EDGE_AVAILABLE


def _clean_text(text: str) -> str:
    """去除 markdown 标记与剔除占位符，让语音念起来自然。"""
    return (
        (text or "")
        .replace("**", "")
        .replace("#", "")
        .replace("[来源", "来源")
        .replace("（已按禁忌剔除）", "已剔除")
        .strip()
    )


async def text_to_speech(text: str, voice: str = VOICE_DEFAULT) -> bytes:
    """文本转 MP3 字节流；edge-tts 缺失或合成失败时抛异常（路由统一 503 降级）。"""
    if not _EDGE_AVAILABLE:
        raise RuntimeError("edge-tts unavailable")
    communicate = edge_tts.Communicate(_clean_text(text), voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    data = buf.getvalue()
    if not data:
        raise RuntimeError("empty audio")
    return data
