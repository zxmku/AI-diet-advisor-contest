"""绘制 HealthPick 产品级应用图标（Pillow 纯代码，零外部依赖）。
设计：莫兰迪绿渐变圆角方块 + 白色餐盘 + 盘中绿叶 + 香槟金细环
输出：assets/healthpick_icon.png (1024) + assets/healthpick_icon.ico (多尺寸)
"""
from PIL import Image, ImageDraw, ImageFilter, ImageOps

S = 1024

def lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

# ── 1) 渐变圆角背景 ──────────────────────────────
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
TOP, BOT = (0x4F, 0xA8, 0x7E), (0x1E, 0x5A, 0x3D)  # 浅绿→深绿
for y in range(S):
    t = y / S
    d.line([(0, y), (S, y)], fill=lerp(TOP, BOT, t) + (255,))

mask = Image.new("L", (S, S), 0)
md = ImageDraw.Draw(mask)
md.rounded_rectangle([0, 0, S - 1, S - 1], radius=200, fill=255)
img.putalpha(mask)

# ── 2) 金色细环（盘子外圈点缀）──────────────────
ring_d = ImageDraw.Draw(img)
GOLD = (0xC9, 0xA8, 0x6A)
for r in (368, 358):  # 双圈形成细环
    ring_d.ellipse([S/2 - r, S/2 - r, S/2 + r, S/2 + r], outline=GOLD + (255,), width=6)

# ── 3) 白色餐盘 ──────────────────────────────────
plate_cy = S * 0.50
# 盘沿（白）：外 300 内 215
d.ellipse([S/2 - 300, plate_cy - 300, S/2 + 300, plate_cy + 300], fill=(255, 255, 255, 255))
# 盘心（浅绿渐变）：外 215
d.ellipse([S/2 - 215, plate_cy - 215, S/2 + 215, plate_cy + 215],
          fill=(0xE8, 0xF3, 0xEC, 255))
# 盘心内圈阴影过渡
d.ellipse([S/2 - 205, plate_cy - 205, S/2 + 205, plate_cy + 205],
          fill=(0xDD, 0xEE, 0xE4, 255))

# ── 4) 盘中绿叶苗（茎 + 两片叶 + 顶芽）───────────
leaf_d = ImageDraw.Draw(img)
STEM = (0x2E, 0x8B, 0x57, 255)
LEAF = (0x34, 0x9E, 0x67, 255)
LEAF_D = (0x27, 0x7A, 0x4E, 255)
cx, cy = S / 2, plate_cy

def leaf(cx2, cy2, w, h, ang, fill):
    lf = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lf)
    ld.ellipse([cx2 - w / 2, cy2 - h / 2, cx2 + w / 2, cy2 + h / 2], fill=fill)
    lf = lf.rotate(ang, center=(cx2, cy2), resample=Image.BICUBIC)
    img.alpha_composite(lf)

# 茎：从盘底到盘心上方
leaf_d.line([(cx, cy + 95), (cx, cy - 10)], fill=STEM, width=26)
# 左叶
leaf(cx - 52, cy + 22, 200, 120, -38, LEAF)
# 右叶
leaf(cx + 52, cy + 22, 200, 120, 38, LEAF)
# 顶芽（小圆）
leaf_d.ellipse([cx - 26, cy - 48, cx + 26, cy + 4], fill=LEAF_D)
# 叶脉：左叶主脉
leaf_d.line([(cx - 30, cy + 62), (cx - 118, cy + 2)], fill=LEAF_D, width=10)
leaf_d.line([(cx + 30, cy + 62), (cx + 118, cy + 2)], fill=LEAF_D, width=10)

# 高光：左上轻微光晕
glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.ellipse([S*0.14, S*0.10, S*0.48, S*0.42], fill=(255, 255, 255, 26))
glow = glow.filter(ImageFilter.GaussianBlur(60))
img.alpha_composite(glow)

# ── 5) 输出 ──────────────────────────────────────
import os
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "healthpick_icon.png")
ico_path = os.path.join(out_dir, "healthpick_icon.ico")
img.save(png_path, "PNG")
# 多尺寸 ICO（Windows 图标标准尺寸）
sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
img.save(ico_path, "ICO", sizes=sizes)
print("ICON_DONE")
print("PNG:", png_path)
print("ICO:", ico_path)
