from pathlib import Path
from PIL import Image, ImageDraw


ROOT = Path('/data/ZhaoX/BoxFusion')
BASE = ROOT / 'figure_assets/carr3d_framework_real_image_edit_20261107/carr3d_framework_real_images_only.png'
READOUT = ROOT / 'figure_assets/online_detection_readout_real_dual_branch_20261107/online_detection_readout_dual_white.png'
OUT = ROOT / 'figure_assets/carr3d_framework_real_image_edit_20261107/carr3d_framework_exact_readout.png'


def arrow(draw, points, color, width=4, head=11):
    draw.line(points, fill=color, width=width, joint='curve')
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    back_x, back_y = x2 - head * ux, y2 - head * uy
    wing = head * 0.48
    draw.polygon(
        [(x2, y2), (back_x + wing * px, back_y + wing * py),
         (back_x - wing * px, back_y - wing * py)],
        fill=color,
    )


canvas = Image.open(BASE).convert('RGB')
source = Image.open(READOUT).convert('RGB')

# Replace only the photographic result area. The title, caption, panel border and
# all upstream workflow content remain pixel-identical to the base figure.
region = (1782, 125, 2165, 584)
draw = ImageDraw.Draw(canvas)
draw.rectangle(region, fill='white')

max_w, max_h = 372, 420
scale = min(max_w / source.width, max_h / source.height)
new_size = (round(source.width * scale), round(source.height * scale))
source = source.resize(new_size, Image.Resampling.LANCZOS)
x = region[0] + (region[2] - region[0] - new_size[0]) // 2
y = region[1] + (region[3] - region[1] - new_size[1]) // 2
canvas.paste(source, (x, y))

# Restore the three readout connectors at the panel boundary. Keeping the
# arrowheads outside the raster makes the embedded source image unobstructed.
draw = ImageDraw.Draw(canvas)
arrow(draw, [(1738, 223), (1784, 223)], '#8122e8', 4, 12)
arrow(draw, [(1738, 420), (1784, 420)], '#ff5a12', 4, 12)
arrow(draw, [(1738, 603), (1784, 603)], '#f1c400', 4, 12)

canvas.save(OUT, dpi=(300, 300))
print(OUT)
