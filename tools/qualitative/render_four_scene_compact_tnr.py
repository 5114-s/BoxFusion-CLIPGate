#!/usr/bin/env python3
"""Render the audited four-scene comparison with Times New Roman typography."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.qualitative import render_four_scene_compact as compact


OUT = ROOT / (
    "figure_assets/"
    "qualitative_scannet_ca1m_onlineanyseg_four_columns_compact_tnr_20261113"
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    compact.OUT = OUT
    compact.source.OUT = OUT
    compact.source.opt.OUT = OUT

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Liberation Serif",
                "serif",
            ],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": compact.source.opt.INK,
        }
    )

    renderer = compact.source.original.load_renderer()
    payloads, audits, hashes = compact.source.load_payloads(renderer)
    exports = compact.render(payloads, renderer)
    effective_font = str(Path(findfont(FontProperties(family="serif"))))

    manifest = {
        "schema": "carr3d.qualitative.four_columns.compact.tnr.v1",
        "layout_only_change": True,
        "methods": list(compact.TITLES),
        "scenes": audits,
        "input_sha256": hashes,
        "exports": exports,
        "prediction_geometry_modified": False,
        "typography": {
            "requested": "Times New Roman",
            "family_stack": list(plt.rcParams["font.serif"]),
            "host_raster_font": effective_font,
            "editable_svg_text": True,
        },
        "spacing": {
            "figure_height_inches": 5.05,
            "wspace": -0.025,
            "hspace": -0.105,
            "legend_and_footer_omitted": True,
        },
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "README.txt").write_text(
        "Times New Roman typography edition of the audited four-scene comparison.\n"
        "Geometry, selections, colors, cameras, viewports, spacing, and font sizes are unchanged.\n"
        "The editable SVG keeps Times New Roman first in its live font-family stack.\n",
        encoding="utf-8",
    )

    bundle = OUT.with_suffix(".zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, f"{OUT.name}/{path.relative_to(OUT)}")

    print(
        json.dumps(
            {
                "output": str(OUT),
                "exports": exports,
                "bundle": str(bundle),
                "host_raster_font": effective_font,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
