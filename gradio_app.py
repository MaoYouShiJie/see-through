import os
import os.path as osp
import sys
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "common"))

from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr
from utils.io_utils import load_parts

VALID_BODY_PARTS_V2 = [
    "hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear",
    "nose", "mouth", "neck", "neckwear", "topwear", "handwear",
    "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
]

HEAD_TAGS_V3 = [
    "headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash",
    "eyewear", "ears", "earwear", "nose", "mouth",
]

BODY_TAGS_V3 = [
    "front hair", "back hair", "head", "neck", "neckwear", "topwear",
    "handwear", "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
]

def ensure_assets_symlink():
    root = osp.dirname(osp.abspath(__file__))
    assets_link = osp.join(root, "assets")
    if not osp.exists(assets_link):
        common_assets = osp.join(root, "common", "assets")
        if osp.exists(common_assets):
            try:
                os.symlink(common_assets, assets_link)
            except OSError:
                shutil.copytree(common_assets, assets_link, dirs_exist_ok=True)

def collect_layer_images(saved_dir):
    opt_dir = osp.join(saved_dir, "optimized")
    layer_files = []
    if osp.exists(opt_dir):
        for f in sorted(os.listdir(opt_dir)):
            if f.lower().endswith(".png"):
                layer_files.append(osp.join(opt_dir, f))
    if not layer_files:
        for f in sorted(os.listdir(saved_dir)):
            if f.lower().endswith(".png") and f not in ("src_img.png", "src_head.png", "depth.png"):
                layer_files.append(osp.join(saved_dir, f))
    return layer_files

def run_pipeline(
    input_image,
    resolution,
    inference_steps,
    group_offload,
    save_to_psd,
    progress=gr.Progress(),
):
    ensure_assets_symlink()

    save_dir = osp.join(osp.dirname(osp.abspath(__file__)), "workspace", "layerdiff_output")
    os.makedirs(save_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        src_path = f.name
        input_image.save(src_path)

    srcname = osp.splitext(osp.basename(src_path))[0]
    saved = osp.join(save_dir, srcname)

    try:
        progress(0.05, desc="Running LayerDiff 3D (transparent layer generation)...")
        apply_layerdiff(
            src_path,
            "layerdifforg/seethroughv0.0.2_layerdiff3d",
            save_dir=save_dir,
            resolution=resolution,
            num_inference_steps=inference_steps,
            group_offload=group_offload,
            disable_progressbar=True,
        )

        progress(0.45, desc="Estimating depth (Marigold)...")
        apply_marigold(
            src_path,
            "24yearsold/seethroughv0.0.1_marigold",
            save_dir=save_dir,
            resolution=768,
            group_offload=group_offload,
            disable_progressbar=True,
        )

        progress(0.75, desc="Extracting and optimizing layers...")
        further_extr(saved, rotate=False, save_to_psd=save_to_psd, tblr_split=True)

        progress(0.9, desc="Collecting results...")
        layer_paths = collect_layer_images(saved)
        layer_images = []
        layer_labels = []
        for lp in layer_paths:
            label = osp.splitext(osp.basename(lp))[0]
            try:
                img = Image.open(lp).convert("RGBA")
                layer_images.append(img)
                layer_labels.append(label)
            except Exception:
                pass

        psd_path = osp.join(save_dir, f"{srcname}.psd")
        psd_available = osp.exists(psd_path)

        src_img = Image.open(src_path).convert("RGBA")

        progress(1.0, desc="Done!")

        gallery_items = [
            (img, label) for img, label in zip(layer_images, layer_labels)
        ]

        return (
            src_img,
            gallery_items,
            psd_path if psd_available else None,
            f"Success! Generated {len(layer_images)} layers.",
        )

    except Exception as e:
        return (
            None,
            [],
            [],
            None,
            f"Error: {str(e)}",
        )
    finally:
        if osp.exists(src_path):
            os.unlink(src_path)

css = """
.gallery-container { min-height: 300px; }
.output-image { border-radius: 8px; }
.title-text { text-align: center; font-size: 1.5em; margin-bottom: 0.5em; }
"""

with gr.Blocks(
    css=css,
    title="See-Through: Anime Layer Decomposition",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # See-Through: Anime Layer Decomposition
        **Single-image layer decomposition for anime characters** — SIGGRAPH 2026  
        Upload an anime character image to decompose it into up to 23 semantic layers.

        > ⚠️ Requires ~12-16 GB VRAM at 1280 resolution. Enable **Group Offload** for 10 GB GPUs.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Input Image",
                height=400,
            )

            with gr.Accordion("Advanced Settings", open=False):
                resolution = gr.Slider(
                    minimum=512, maximum=1536, value=1280, step=64,
                    label="LayerDiff Resolution",
                )
                inference_steps = gr.Slider(
                    minimum=10, maximum=60, value=30, step=1,
                    label="Inference Steps",
                )
                group_offload = gr.Checkbox(
                    label="Group Offload (reduces VRAM, slower)",
                    value=False,
                )
                save_to_psd = gr.Checkbox(
                    label="Export PSD file",
                    value=True,
                )

            run_btn = gr.Button("Run Decomposition", variant="primary", size="lg")
            status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            original_output = gr.Image(
                type="pil",
                label="Original Image",
                height=300,
            )

            psd_download = gr.File(
                label="Download PSD",
                visible=True,
            )

    gr.Markdown("### Decomposed Layers")
    layer_gallery = gr.Gallery(
        label="Layer Preview",
        columns=4,
        rows=2,
        height=400,
        object_fit="contain",
        allow_preview=True,
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            input_image,
            resolution,
            inference_steps,
            group_offload,
            save_to_psd,
        ],
        outputs=[
            original_output,
            layer_gallery,
            psd_download,
            status_text,
        ],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
