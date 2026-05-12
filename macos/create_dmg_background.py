#!/usr/bin/env python3
"""
Generate the Kura Medical DMG installer background image.
Minimal layout: subtle gradient, one arrow, one instruction line.
Outputs:  macos/assets/dmg_background.png      (660×420  – 1x)
          macos/assets/dmg_background@2x.png   (1320×840 – retina)
"""

from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1320, 840

BG_TOP    = (248, 250, 253)
BG_BOT    = (232, 238, 247)
ARROW     = (100, 116, 139)
TEXT_MID  = (100, 116, 139)

# create-dmg: --icon "Kura.app" 160 200  --icon "Programme" 490 200
APP_CX,  APP_CY  = 320, 400
DEST_CX, DEST_CY = 980, 400
ICON_HALF = 100

HN = "/System/Library/Fonts/HelveticaNeue.ttc"
HN_LIGHT = 7

def font(size, face=HN_LIGHT):
    try:
        return ImageFont.truetype(HN, size, index=face)
    except Exception:
        return ImageFont.load_default()

def gradient(draw, w, h, top, bot):
    for y in range(h):
        t = y / max(h - 1, 1)
        c = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=c)

img  = Image.new("RGBA", (W, H), BG_TOP)
draw = ImageDraw.Draw(img)
gradient(draw, W, H, BG_TOP, BG_BOT)

# Arrow between the two icons
ARR_LEFT  = APP_CX + ICON_HALF + 48
ARR_RIGHT = DEST_CX - ICON_HALF - 48
ARR_Y     = APP_CY

draw.line([(ARR_LEFT, ARR_Y), (ARR_RIGHT - 20, ARR_Y)], fill=ARROW, width=5)
ah = 20
draw.polygon([
    (ARR_RIGHT - 22, ARR_Y - ah),
    (ARR_RIGHT,      ARR_Y),
    (ARR_RIGHT - 22, ARR_Y + ah),
], fill=ARROW)

# Single instruction line below the icons
f_label = font(30)
LABEL   = "Kura in den Programme-Ordner ziehen"
lw      = draw.textbbox((0, 0), LABEL, font=f_label)[2]
lx      = (W - lw) // 2
draw.text((lx, APP_CY + ICON_HALF + 140), LABEL, font=f_label, fill=TEXT_MID)

out_dir = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(out_dir, exist_ok=True)

p2 = os.path.join(out_dir, "dmg_background@2x.png")
img.convert("RGB").save(p2, "PNG")
print(f"  [bg] {p2}  ({W}x{H})")

p1 = os.path.join(out_dir, "dmg_background.png")
img.convert("RGB").resize((W // 2, H // 2), Image.LANCZOS).save(p1, "PNG")
print(f"  [bg] {p1}  ({W//2}x{H//2})")