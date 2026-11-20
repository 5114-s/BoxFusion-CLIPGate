#!/usr/bin/env python3
"""Build the compact, code-faithful ReCaR-3D overview used in the paper.

Raster photographs/maps are real assets from scene0432_01.  Diagram geometry,
labels and arrows remain vector objects in the SVG export.
"""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure_assets/recar3d_full_flowchart_optimized_20261103"
W, H = 5200, 2100

INK = "#132238"
MUTED = "#53657b"
GRAY = "#758396"
PANEL = "#fbfcfe"
BLUE = "#1769e8"
BLUE_LIGHT = "#eef6ff"
PURPLE = "#8624e8"
PURPLE_LIGHT = "#f7f0ff"
ORANGE = "#fb5a16"
ORANGE_LIGHT = "#fff3ec"
AMBER = "#c48300"
AMBER_LIGHT = "#fff7dd"
GREEN = "#0fa95b"
GREEN_LIGHT = "#effbf5"

SRC = ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01"
RGBD = ROOT / "figure_assets/rgbd_input_scannet_scene0432_01"
NATIVE = ROOT / "figure_assets/native_online_mapper_corrected_scene0432_01_20261103"

ASSETS = {
    "rgbd0": RGBD / "frame_0000_rgbd_inset.png",
    "rgbd1": RGBD / "frame_0144_rgbd_inset.png",
    "rgbd2": RGBD / "frame_0288_rgbd_inset.png",
    "candidate": SRC / "02c_reliable_view_top3_frame0050.png",
    "view1": SRC / "02a_reliable_view_top1_frame0000.png",
    "view2": SRC / "02b_reliable_view_top2_frame0025.png",
    "view3": SRC / "02c_reliable_view_top3_frame0050.png",
    "native_map": NATIVE / "02_updated_native_map_kf0050.png",
    "lifted_proposals": SRC / "09b_boxer_lifted_proposals_orange.png",
    "lifted_anchors": SRC / "09d_boxer_lifted_anchors_amber.png",
    "hcsr_matches": SRC / "06b_m2_match_cached_proposals_triptych_ottoman.png",
    "final_map": SRC / "07_final_3d_map_figure.png",
}


def font(size: int, bold: bool = False, italic: bool = False):
    if bold:
        name = "DejaVuSans-Bold.ttf"
    elif italic:
        name = "DejaVuSans-Oblique.ttf"
    else:
        name = "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def image_uri(path: Path) -> str:
    mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class Figure:
    def __init__(self):
        self.image = Image.new("RGBA", (W, H), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg: list[str] = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="arrow-gray" markerWidth="16" markerHeight="16" refX="14" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 Z" fill="{GRAY}"/></marker>
  <marker id="arrow-blue" markerWidth="16" markerHeight="16" refX="14" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 Z" fill="{BLUE}"/></marker>
  <marker id="arrow-purple" markerWidth="16" markerHeight="16" refX="14" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 Z" fill="{PURPLE}"/></marker>
  <marker id="arrow-orange" markerWidth="16" markerHeight="16" refX="14" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 Z" fill="{ORANGE}"/></marker>
  <marker id="arrow-amber" markerWidth="16" markerHeight="16" refX="14" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 Z" fill="{AMBER}"/></marker>
  <marker id="arrow-green" markerWidth="16" markerHeight="16" refX="14" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 Z" fill="{GREEN}"/></marker>
  <style>text {{font-family:Arial,Helvetica,sans-serif;}}</style>
</defs>''']
        self.clip_count = 0

    def rrect(self, box, radius, fill, outline=None, width=3):
        x1, y1, x2, y2 = box
        self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        stroke = outline if outline else "none"
        self.svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def line(self, points, fill=GRAY, width=8, dash=None, arrow=False):
        self.draw.line(points, fill=fill, width=width, joint="curve")
        if dash:
            # Cover solid line, then redraw dash sequence in PIL.
            self.draw.line(points, fill="white", width=width + 2, joint="curve")
            self._dashed_pil(points, fill, width, dash[0], dash[1])
        if arrow:
            self._head(points[-2], points[-1], fill, width)
        pts = " ".join(f"{x},{y}" for x, y in points)
        d = f' stroke-dasharray="{dash[0]} {dash[1]}"' if dash else ""
        marker = f' marker-end="url(#arrow-{self._marker(fill)})"' if arrow else ""
        self.svg.append(f'<polyline points="{pts}" fill="none" stroke="{fill}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{d}{marker}/>')

    def _marker(self, color):
        return {GRAY:"gray", BLUE:"blue", PURPLE:"purple", ORANGE:"orange", AMBER:"amber", GREEN:"green"}.get(color, "gray")

    def _head(self, a, b, color, width):
        x1, y1 = a; x2, y2 = b
        ang = math.atan2(y2-y1, x2-x1)
        size = max(22, width * 2.7)
        wing = size * .62
        p1 = (x2, y2)
        p2 = (x2-size*math.cos(ang)+wing*math.sin(ang), y2-size*math.sin(ang)-wing*math.cos(ang))
        p3 = (x2-size*math.cos(ang)-wing*math.sin(ang), y2-size*math.sin(ang)+wing*math.cos(ang))
        self.draw.polygon((p1,p2,p3), fill=color)

    def _dashed_pil(self, points, color, width, dash, gap):
        for a, b in zip(points, points[1:]):
            x1,y1=a; x2,y2=b
            length=math.hypot(x2-x1,y2-y1)
            if not length: continue
            ux,uy=(x2-x1)/length,(y2-y1)/length
            pos=0.0
            while pos<length:
                end=min(length,pos+dash)
                self.draw.line((x1+ux*pos,y1+uy*pos,x1+ux*end,y1+uy*end),fill=color,width=width)
                pos += dash+gap

    def polygon(self, points, fill, outline=None, width=3):
        self.draw.polygon(points, fill=fill)
        if outline:
            self.draw.line(points+[points[0]], fill=outline, width=width, joint="curve")
        pts=" ".join(f"{x},{y}" for x,y in points)
        self.svg.append(f'<polygon points="{pts}" fill="{fill}" stroke="{outline or "none"}" stroke-width="{width}"/>')

    def ellipse(self, box, fill, outline=None, width=3):
        self.draw.ellipse(box, fill=fill, outline=outline, width=width)
        x1,y1,x2,y2=box
        self.svg.append(f'<ellipse cx="{(x1+x2)/2}" cy="{(y1+y2)/2}" rx="{(x2-x1)/2}" ry="{(y2-y1)/2}" fill="{fill}" stroke="{outline or "none"}" stroke-width="{width}"/>')

    def text(self, xy, value, size, fill=INK, bold=False, italic=False, anchor="mm"):
        x,y=xy
        f=font(size,bold,italic)
        lines=value.split("\n")
        spacing=int(size*.22)
        b=self.draw.multiline_textbbox((0,0),value,font=f,spacing=spacing,align="center")
        tw,th=b[2]-b[0],b[3]-b[1]
        if anchor=="mm": px,py=x-tw/2,y-th/2
        elif anchor=="lm": px,py=x,y-th/2
        elif anchor=="mt": px,py=x-tw/2,y
        else: px,py=x,y
        self.draw.multiline_text((px,py),value,font=f,fill=fill,spacing=spacing,align="center")
        weight="700" if bold else "400"
        style="italic" if italic else "normal"
        svg_anchor="middle" if anchor in ("mm","mt") else "start"
        # SVG multiline baseline is positioned to match the centered PIL block.
        start_y = y - (len(lines)-1)*(size+spacing)/2 + size*.34 if anchor in ("mm","lm") else y+size*.8
        for i,line in enumerate(lines):
            self.svg.append(f'<text x="{x}" y="{start_y+i*(size+spacing)}" text-anchor="{svg_anchor}" font-size="{size}" font-weight="{weight}" font-style="{style}" fill="{fill}">{html.escape(line)}</text>')

    def image_box(self, path: Path, box, radius=20, contain=False, border="#c5d0dc", width=3, bg="white"):
        x,y,w,h=box
        src=Image.open(path).convert("RGBA")
        scale=min(w/src.width,h/src.height) if contain else max(w/src.width,h/src.height)
        nw,nh=max(1,round(src.width*scale)),max(1,round(src.height*scale))
        src=src.resize((nw,nh),Image.Resampling.LANCZOS)
        panel=Image.new("RGBA",(w,h),bg)
        panel.alpha_composite(src,((w-nw)//2,(h-nh)//2))
        mask=Image.new("L",(w,h),0)
        ImageDraw.Draw(mask).rounded_rectangle((0,0,w-1,h-1),radius=radius,fill=255)
        self.image.paste(panel,(x,y),mask)
        self.draw.rounded_rectangle((x,y,x+w,y+h),radius=radius,outline=border,width=width)
        self.clip_count+=1; cid=f"clip{self.clip_count}"
        fit="meet" if contain else "slice"
        self.svg.append(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}"/></clipPath>')
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{bg}" stroke="{border}" stroke-width="{width}"/>')
        self.svg.append(f'<image href="{image_uri(path)}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid {fit}" clip-path="url(#{cid})"/>')

    def cuboid(self,x,y,w,h,dep,color,fill="none",width=5):
        dx,dy=dep,-int(dep*.58)
        faces=[[(x,y),(x+w,y),(x+w,y+h),(x,y+h)],[(x,y),(x+dx,y+dy),(x+w+dx,y+dy),(x+w,y)],[(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)]]
        for p in faces:
            self.polygon(p,fill if fill!="none" else "#ffffff",color,width)

    def save(self):
        OUT.mkdir(parents=True,exist_ok=True)
        png=OUT/"recar3d_full_flowchart_optimized.png"
        svg=OUT/"recar3d_full_flowchart_optimized.svg"
        pdf=OUT/"recar3d_full_flowchart_optimized.pdf"
        self.image.convert("RGB").save(png,dpi=(600,600),quality=96)
        self.svg.append("</svg>")
        svg.write_text("\n".join(self.svg),encoding="utf-8")
        self.image.convert("RGB").save(pdf,"PDF",resolution=300.0)
        return png,svg,pdf


def camera_icon(fig: Figure, x, y, scale=1.0):
    fig.ellipse((x-13*scale,y-13*scale,x+13*scale,y+13*scale),GREEN,"white",3)
    fig.line([(x,y-18*scale),(x-34*scale,y-76*scale),(x+34*scale,y-76*scale),(x,y-18*scale)],GREEN,max(3,int(4*scale)))


def draw_voxel(fig: Figure,x,y,size=210):
    fig.cuboid(x,y,size,size,58,"#7d98b8","#f8fbff",4)
    for i in range(1,4):
        xx=x+size*i/4; yy=y+size*i/4
        fig.line([(xx,y),(xx,y+size)],"#94a9c1",3)
        fig.line([(x,yy),(x+size,yy)],"#94a9c1",3)


def build():
    missing=[str(p) for p in ASSETS.values() if not p.exists()]
    if missing: raise FileNotFoundError("Missing assets:\n"+"\n".join(missing))
    f=Figure()

    # Input stream.
    f.rrect((25,55,605,2045),45,GREEN_LIGHT,GREEN,7)
    f.text((315,120),"Posed RGB-D Stream",46,INK,True)
    for i,key in enumerate(("rgbd0","rgbd1","rgbd2")):
        f.image_box(ASSETS[key],(65,180+i*430,500,360),22,False,"#b9d9c8",3)
    f.text((315,1450),"⋮",68,MUTED,True)
    for i,x in enumerate((175,315,455)):
        camera_icon(f,x,1755+25*(i%2),1.2)
        if i<2: f.line([(x+20,1765),(x+115,1790)],"#476a5b",5,(7,8))
    f.text((315,1915),"RGB · depth · pose",32,MUTED)

    # Native mapper.
    f.rrect((665,55,3140,820),45,BLUE_LIGHT,BLUE,7)
    f.text((1850,105),"Inherited Native Online Mapper",57,INK,True)
    f.rrect((2760,78,3065,145),20,"#e4f0fd","#8db5df",3)
    f.text((2912,111),"Parameter-frozen",29,BLUE,True)
    f.text((920,190),"Current Candidate",34,INK,True)
    f.image_box(ASSETS["candidate"],(710,225,420,330),18,False,"#b4cae5",3)
    f.line([(1130,390),(1210,390)],GRAY,10,arrow=True)
    f.rrect((1225,265,1585,515),25,"#e6f1fd","#82acd9",5)
    f.text((1405,345),"Native\nAssociation",37,INK,True)
    f.text((1405,462),"3D NMS · 2D matching",24,MUTED)
    f.line([(1585,390),(1660,390)],GRAY,10,arrow=True)
    f.text((2025,190),"Associated Observation History",34,INK,True)
    for i,key in enumerate(("view1","view2","view3")):
        x=1655+i*245
        f.image_box(ASSETS[key],(x,245,230,250),15,False,"#b4cae5",3)
        f.ellipse((x+6,250,x+58,302),"#50677e","white",3)
        f.text((x+32,276),str(i+1),27,"white",True)
    f.text((2020,545),"Reliable-View Top-K (K=3)",27,MUTED,True)
    f.line([(2390,390),(2460,390)],GRAY,10,arrow=True)
    f.rrect((2475,290,2725,490),25,"#e6f1fd","#82acd9",5)
    f.text((2600,390),"PFO\nFusion",38,INK,True)
    f.line([(2725,390),(2785,390)],GRAY,10,arrow=True)
    f.image_box(ASSETS["native_map"],(2790,195,305,390),18,True,"#b4cae5",3)
    f.text((2942,640),"Native Map N⁽ᵗ⁾",34,BLUE,True)
    f.text((1850,755),"Frozen native association and geometry fusion",25,MUTED,italic=True)

    # HCSR.
    f.rrect((3200,55,5175,820),45,PURPLE_LIGHT,PURPLE,7)
    f.text((4187,105),"HCSR | Historical Candidate-Support Reranking",54,INK,True)
    f.text((3510,190),"Project & Match",34,INK,True)
    # The source is a real three-view horizontal triptych.  Preserve its full
    # extent; cover-cropping would incorrectly hide the first/third views.
    f.image_box(ASSETS["hcsr_matches"],(3240,245,620,180),18,True,"#d5c1ed",3)
    f.text((3550,475),"blue: native projection · purple: matched proposal",23,MUTED)
    f.line([(3860,390),(3915,390)],PURPLE,10,arrow=True)
    f.text((4210,190),"Historical Support",34,INK,True)
    # Support bars and running maximum.
    base_y=525; xs=[3965,4050,4135,4220,4305,4390]
    vals=[80,135,95,170,145,215]
    for x,v in zip(xs,vals):
        f.rrect((x,base_y-v,x+48,base_y),8,"#dcbaf8",None,0)
    levels=[]; cur=0
    for v in vals: cur=max(cur,v); levels.append(cur)
    pts=[]
    for i,(x,lvl) in enumerate(zip(xs,levels)):
        y=base_y-lvl-28
        if i==0: pts.append((x,y))
        else: pts.extend([(x,y_prev),(x,y)])
        pts.append((x+48,y)); y_prev=y
    f.line(pts,PURPLE,7)
    for x,lvl in zip(xs,levels): f.ellipse((x+16,base_y-lvl-40,x+32,base_y-lvl-24),"white",PURPLE,5)
    f.text((4205,595),"running maximum over observed views",24,PURPLE,True)
    f.line([(4460,390),(4505,390)],PURPLE,10,arrow=True)
    f.text((4820,190),"Native Reranking",34,INK,True)
    for i,(lab,x) in enumerate(zip("ABC",(4540,4705,4870))):
        f.cuboid(x,270,75,65,25,BLUE,"#eef6ff",4); f.text((x+38,304),lab,25,INK,True)
    f.text((4760,245),"Before: A > B > C",25,INK,True)
    f.line([(4760,350),(4760,395)],PURPLE,8,arrow=True)
    for lab,x in zip("CAB",(4540,4705,4870)):
        f.cuboid(x,445,75,65,25,BLUE,"#eef6ff",4); f.text((x+38,479),lab,25,INK,True)
    f.text((4760,580),"After: C > A > B",25,INK,True)
    f.text((4187,750),"Only native scores are updated; identities and geometry remain fixed.",25,MUTED,italic=True)
    f.line([(3140,390),(3200,390)],GRAY,10,arrow=True)

    # Multi-level candidate recovery.
    f.rrect((665,900,3780,2045),45,"#fffaf7",ORANGE,7)
    f.text((2220,950),"Multi-Level Candidate Recovery",55,INK,True)
    f.rrect((700,1010,1745,1995),35,"#fffdfb","#d8a78e",4)
    f.text((1222,1060),"Frozen Auxiliary Evidence",41,INK,True)
    f.rrect((720,1160,960,1780),25,"#f2f5f8","#728196",4)
    f.text((840,1390),"WeDetect-Uni",30,INK,True)
    f.text((840,1450),"parameter-frozen",20,MUTED)
    f.rrect((995,1190,1170,1415),22,ORANGE_LIGHT,ORANGE,4)
    f.text((1082,1302),"2D NMS",29,ORANGE,True)
    f.rrect((995,1530,1170,1755),22,AMBER_LIGHT,AMBER,4)
    f.text((1082,1642),"Top-300",29,AMBER,True)
    f.rrect((1205,1240,1415,1710),25,"#f2f5f8","#728196",4)
    f.text((1310,1430),"Frozen\nBoxer",31,INK,True)
    f.text((1310,1535),"2D → 3D",22,MUTED)
    f.image_box(ASSETS["lifted_proposals"],(1450,1125,265,300),15,False,"#ffc5ad",3)
    f.image_box(ASSETS["lifted_anchors"],(1450,1510,265,300),15,False,"#e7c36b",3)
    f.text((1582,1465),"Lifted Proposals",24,ORANGE,True)
    f.text((1582,1850),"Lifted Anchors",24,AMBER,True)
    f.line([(960,1302),(995,1302)],ORANGE,8,arrow=True)
    f.line([(1170,1302),(1205,1302)],ORANGE,8,arrow=True)
    f.line([(1415,1302),(1450,1302)],ORANGE,8,arrow=True)
    f.line([(960,1642),(995,1642)],AMBER,8,arrow=True)
    f.line([(1170,1642),(1205,1642)],AMBER,8,arrow=True)
    f.line([(1415,1642),(1450,1642)],AMBER,8,arrow=True)

    # PLR panel.
    f.rrect((1815,1010,3740,1485),30,ORANGE_LIGHT,ORANGE,5)
    f.text((2775,1050),"PLR | Proposal-Level Recovery",40,INK,True)
    f.text((2110,1125),"Track + 3-Frame Verify",29,INK,True)
    for i,x in enumerate((1910,2050,2190),1):
        f.text((x+35,1180),f"t{i}",22,INK,italic=True)
        f.cuboid(x,1210,65,58,24,ORANGE,"#fff7f2",4)
        f.ellipse((x+23,1325,x+39,1341),INK)
    f.line([(1885,1333),(2315,1333)],INK,5,arrow=True)
    f.line([(2340,1260),(2425,1260)],GRAY,8,arrow=True)
    f.text((2610,1125),"AABB-IoU Medoid",29,INK,True)
    f.cuboid(2470,1225,90,76,30,"#ffb18f","#fff7f2",3)
    f.cuboid(2520,1205,95,80,32,"#ffcbb7","#fff7f2",3)
    f.cuboid(2600,1235,105,85,36,ORANGE,"#fff0e8",6)
    f.line([(2735,1260),(2815,1260)],ORANGE,8,arrow=True)
    f.text((3160,1125),"Native Check + Ranked Dedup.",29,INK,True)
    f.cuboid(2860,1225,105,85,36,ORANGE,"#fff0e8",6)
    f.text((3025,1265),"IoU",25,MUTED,True)
    f.cuboid(3135,1225,105,85,36,BLUE,"#eef6ff",6)
    f.text((3188,1370),"map reference",20,MUTED)
    f.line([(3190,1100),(3190,1195)],BLUE,5,(10,9),arrow=True)
    f.text((3190,1080),"Native Map N⁽ᵗ⁾",23,BLUE,True)
    f.text((3375,1265),"✓",62,GREEN,True)
    f.line([(3430,1260),(3740,1260)],ORANGE,10,arrow=True)
    f.text((3580,1215),"PLR outputs",24,ORANGE,True)

    # CALR panel.
    f.rrect((1815,1520,3740,1995),30,AMBER_LIGHT,AMBER,5)
    f.text((2775,1560),"CALR | Causal Anchor-Level Recovery",40,INK,True)
    f.text((2215,1635),"3-Frame Voxel Consensus",29,INK,True)
    draw_voxel(f,2050,1710,190)
    for x,y,c,t in ((2100,1785,BLUE,"t₁"),(2150,1848,"#eeb000","t₂"),(2200,1812,ORANGE,"t₃")):
        f.ellipse((x-13,y-13,x+13,y+13),c); f.text((x+35,y),t,21,INK)
    f.line([(2355,1810),(2510,1810)],GRAY,8,arrow=True)
    f.text((2810,1635),"Highest-Score Exemplar",29,INK,True)
    f.cuboid(2615,1735,300,150,75,AMBER,"#fff6d9",7)
    f.text((2765,1810),"★",55,"#e9a600",True)
    f.line([(3020,1810),(3740,1810)],AMBER,10,arrow=True)
    f.text((3380,1758),"tail-score ranked",22,MUTED,italic=True)
    f.text((3580,1870),"CALR outputs",24,AMBER,True)

    # Cross-panel evidence references.
    f.line([(1582,1125),(1582,865),(3370,865),(3370,820)],PURPLE,8,(20,15),arrow=True)
    f.text((2460,850),"post-NMS proposal evidence",25,PURPLE,True)
    f.line([(2942,585),(2942,850),(3260,850),(3260,1010)],"#687b90",6,(13,11),arrow=True)
    f.text((3085,875),"map reference",22,MUTED,True)

    # Unified output assembly and final result.
    f.rrect((3835,1160,4245,1705),35,"#f4f6f8","#7d8a99",5)
    f.text((4040,1260),"Output\nAssembly",40,INK,True)
    f.text((4040,1445),"score-sorted\nconcatenation",25,MUTED)
    f.ellipse((3890,1550,3920,1580),PURPLE)
    f.ellipse((3995,1550,4025,1580),ORANGE)
    f.ellipse((4100,1550,4130,1580),AMBER)
    f.line([(4810,820),(4810,930),(4040,930),(4040,1160)],PURPLE,10,arrow=True)
    f.text((4510,905),"reranked native records",25,PURPLE,True)
    f.line([(3740,1260),(3835,1260)],ORANGE,10,arrow=True)
    f.line([(3740,1810),(3810,1810),(3810,1600),(3835,1600)],AMBER,10,arrow=True)
    f.line([(4245,1435),(4315,1435)],GREEN,12,arrow=True)

    f.rrect((4315,900,5175,2045),45,GREEN_LIGHT,GREEN,7)
    f.text((4745,960),"3D Object Detections",45,INK,True)
    f.image_box(ASSETS["final_map"],(4380,1040,730,790),24,True,"#b6d9c6",3,"#f8fffb")
    f.text((4745,1900),"native + PLR + CALR",29,MUTED,True)
    f.text((4745,1960),"All perception models are parameter-frozen.",22,MUTED,italic=True)

    # Input arrows drawn last for clarity.
    f.line([(605,400),(710,400)],GRAY,10,arrow=True)
    f.line([(605,1520),(720,1520)],GRAY,10,arrow=True)

    png,svg,pdf=f.save()
    manifest={
        "scene":"scene0432_01",
        "semantic_notes":[
            "PFO is separated from Reliable-View Top-K.",
            "The native map is a read-only PLR deduplication reference.",
            "PLR output remains orange after native-map comparison.",
            "The anchor lane explicitly includes Top-300.",
            "Reranked native records, PLR outputs, and CALR outputs converge at one output-assembly node.",
            "Output assembly is score-sorted concatenation with no cross-branch fusion, NMS, or deduplication.",
        ],
        "assets":{k:str(v.relative_to(ROOT)) for k,v in ASSETS.items()},
    }
    (OUT/"source_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    (OUT/"README.md").write_text("""# Optimized ReCaR-3D flowchart

The raster panels use real `scene0432_01` observations and map visualizations. The figure is a method overview: the observation-history thumbnails illustrate the inputs to Reliable-View Top-K and do not claim that an audited PFO receipt stored those exact three view indices. The final map is an illustrative subset; scene geometry is used only for visualization.

The SVG keeps panel borders, arrows, labels, icons, and charts as vector objects. Embedded RGB/map panels remain raster images.

Output assembly follows the runtime implementation: reranked native records, PLR outputs, and CALR outputs are concatenated and score-sorted. No cross-branch association, fusion, unified 3D NMS, or deduplication is applied after concatenation.
""",encoding="utf-8")
    print(png); print(svg); print(pdf)


if __name__=="__main__":
    build()
