#!/usr/bin/env python3
"""
Generate the Kura Medical DMG installer background image.
Outputs:  macos/assets/dmg_background.png      (660×420  – 1x)
          macos/assets/dmg_background@2x.png   (1320×840 – retina)

Run manually:  python3 macos/create_dmg_background.py
Called by:     create_installer.sh  (automatically before build)
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

# ── Canvas (work at 2× retina; downscale for 1×) ────────────────────────────
W, H = 1320, 840

# ── Palette ──────────────────────────────────────────────────────────────────
BG_TOP    = (243, 247, 255)
BG_BOT    = (218, 230, 250)
BLUE      = ( 37,  99, 235)
BLUE_DARK = ( 29,  78, 216)
BLUE_100  = (219, 234, 254)
BLUE_300  = (147, 197, 253)
TEXT_DARK = ( 15,  23,  42)
TEXT_MID  = ( 71,  85, 105)
TEXT_LITE = (180, 195, 215)
RULE      = (203, 213, 225)
WHITE     = (255, 255, 255)

# ── Icon layout (2× pixel coords) ───────────────────────────────────────────
# create-dmg: --icon-size 100  --icon "Kura.app" 160 200  --icon "Install…" 490 200
# (x, y) is icon *centre* in 1× pts; multiply by 2 for this 2× canvas.
APP_CX,  APP_CY  = 320, 400   # Kura.app centre
INST_CX, INST_CY = 980, 400   # Installer centre
ICON_HALF = 100                # 100pt icon-size → ±100px at 2×

# ── Fonts ────────────────────────────────────────────────────────────────────
HN = "/System/Library/Fonts/HelveticaNeue.ttc"
HN_BOLD, HN_REG, HN_LIGHT, HN_MEDIUM = 1, 0, 7, 10

def font(size, face=HN_REG):
    try:
        return ImageFont.truetype(HN, size, index=face)
    except Exception:
        return ImageFont.load_default()

# ── Helpers ───────────────────────────────────────────────────────────────────
def gradient(draw, w, h, top, bot):
    for y in range(h):
        t = y / max(h - 1, 1)
        c = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=c)

def rrect(draw, box, r, fill=None, outline=None, width=3):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

def centered_x(draw, text, f):
    bb = draw.textbbox((0, 0), text, font=f)
    return (W - (bb[2] - bb[0])) // 2

def text_h(draw, text, f):
    bb = draw.textbbox((0, 0), text, font=f)
    return bb[3] - bb[1]

def soft_glow(size, cx, cy, r, color, blur=50):
    g = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(g).rounded_rectangle(
        [cx - r, cy - r, cx + r, cy + r],
        radius=r // 3, fill=(*color, 90),
    )
    return g.filter(ImageFilter.GaussianBlur(radius=blur))

# ── Build ─────────────────────────────────────────────────────────────────────
img  = Image.new("RGBA", (W, H), BG_TOP)
draw = ImageDraw.Draw(img)
gradient(draw, W, H, BG_TOP, BG_BOT)

# ─── Header card ──────────────────────────────────────────────────────────────
f_title = font(64, HN_BOLD)
f_sub   = font(32, HN_LIGHT)

TITLE_TXT = "Kura Medical"
SUB_TXT   = "KI-Assistent für medizinische Transkription"

th = text_h(draw, TITLE_TXT, f_title)
sh = text_h(draw, SUB_TXT,   f_sub)

LOGO_PX   = 68
GAP       = 16   # gap between title and subtitle
V_PAD     = 20   # vertical padding inside card
block_h   = th + GAP + sh
HDR_Y0    = 22
HDR_Y1    = HDR_Y0 + V_PAD * 2 + max(block_h, LOGO_PX)

rrect(draw, [44, HDR_Y0, W - 44, HDR_Y1], r=24, fill=WHITE, outline=RULE, width=2)

ASSETS   = os.path.join(os.path.dirname(__file__), "..", "assets")
logo_img = (
    Image.open(os.path.join(ASSETS, "icon.png"))
    .convert("RGBA")
    .resize((LOGO_PX, LOGO_PX), Image.LANCZOS)
)

# Measure text block width to centre logo + text together
title_w = draw.textbbox((0, 0), TITLE_TXT, font=f_title)[2]
sub_w   = draw.textbbox((0, 0), SUB_TXT,   font=f_sub)[2]
txt_w   = max(title_w, sub_w)
logo_gap = 18
total_w = LOGO_PX + logo_gap + txt_w
BX = (W - total_w) // 2   # left edge of the whole logo+text block

logo_top = HDR_Y0 + (HDR_Y1 - HDR_Y0 - LOGO_PX) // 2
img.paste(logo_img, (BX, logo_top), logo_img)

txt_x     = BX + LOGO_PX + logo_gap
title_top = HDR_Y0 + V_PAD
sub_top   = title_top + th + GAP

draw.text((txt_x, title_top), TITLE_TXT, font=f_title, fill=TEXT_DARK)
draw.text((txt_x, sub_top),   SUB_TXT,   font=f_sub,   fill=TEXT_MID)

# Separator
SEP_Y = HDR_Y1 + 22
draw.line([(80, SEP_Y), (W - 80, SEP_Y)], fill=RULE, width=2)

# ─── Soft glow behind installer zone ─────────────────────────────────────────
img = Image.alpha_composite(
    img, soft_glow((W, H), INST_CX, INST_CY, r=260, color=BLUE_300, blur=60)
)
draw = ImageDraw.Draw(img)

# ─── Installer highlight card ─────────────────────────────────────────────────
# Covers icon zone + macOS filename label (~130px below icon bottom)
CX0 = INST_CX - ICON_HALF - 96
CY0 = INST_CY - ICON_HALF - 20
CX1 = INST_CX + ICON_HALF + 96
CY1 = INST_CY + ICON_HALF + 148

rrect(draw, [CX0, CY0, CX1, CY1], r=32, fill=BLUE_100, outline=BLUE_300, width=4)

# Reassurance lines inside card, below the OS filename label zone
f_helper = font(25, HN_LIGHT)
lines    = [
    "Ein Terminalfenster öffnet sich kurz",
    "und schließt sich automatisch.",
]
for i, line in enumerate(lines):
    lx = INST_CX - (draw.textbbox((0, 0), line, font=f_helper)[2]) // 2
    draw.text((lx, CY1 - 90 + i * 34), line, font=f_helper, fill=TEXT_MID)

# ─── Badge: "DOPPELKLICK ZUM INSTALLIEREN" ────────────────────────────────────
f_badge = font(30, HN_MEDIUM)
BADGE   = "DOPPELKLICK ZUM INSTALLIEREN"
bbb     = draw.textbbox((0, 0), BADGE, font=f_badge)
bw, bh  = bbb[2] - bbb[0], bbb[3] - bbb[1]
BPX, BPY = 28, 14

# Anchor badge so arrow has ~60px of room to the card top
BADGE_BTM = CY0 - 60
BADGE_TOP = BADGE_BTM - bh - BPY * 2
bx        = INST_CX - bw // 2

rrect(draw, [bx - BPX, BADGE_TOP, bx + bw + BPX, BADGE_BTM], r=40, fill=BLUE)
draw.text((bx, BADGE_TOP + BPY), BADGE, font=f_badge, fill=WHITE)

# Arrow badge → card
ARR_CX  = INST_CX
ARR_TOP = BADGE_BTM + 6
ARR_BOT = CY0 - 8
draw.line([(ARR_CX, ARR_TOP), (ARR_CX, ARR_BOT)], fill=BLUE_DARK, width=6)
aw = 20
draw.polygon([
    (ARR_CX - aw, ARR_BOT - 22),
    (ARR_CX + aw, ARR_BOT - 22),
    (ARR_CX,      ARR_BOT + 2),
], fill=BLUE_DARK)

# ─── Left: subtle app label ───────────────────────────────────────────────────
f_lbl = font(28, HN_LIGHT)
lbl   = "Kura Medical App"
lw    = draw.textbbox((0, 0), lbl, font=f_lbl)[2]
draw.text((APP_CX - lw // 2, APP_CY - ICON_HALF - 52), lbl, font=f_lbl, fill=TEXT_LITE)

# ─── Footer ───────────────────────────────────────────────────────────────────
FOOT_Y = H - 82
draw.line([(80, FOOT_Y - 22), (W - 80, FOOT_Y - 22)], fill=RULE, width=2)
f_foot   = font(27, HN_LIGHT)
FOOT_TXT = "Alle Verarbeitung erfolgt lokal auf Ihrem Mac – keine Daten werden ins Internet übertragen."
fx = centered_x(draw, FOOT_TXT, f_foot)
draw.text((fx, FOOT_Y), FOOT_TXT, font=f_foot, fill=TEXT_MID)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(out_dir, exist_ok=True)

p2 = os.path.join(out_dir, "dmg_background@2x.png")
img.convert("RGB").save(p2, "PNG")
print(f"  [bg] {p2}  ({W}x{H})")

p1 = os.path.join(out_dir, "dmg_background.png")
img.convert("RGB").resize((W // 2, H // 2), Image.LANCZOS).save(p1, "PNG")
print(f"  [bg] {p1}  ({W//2}x{H//2})")