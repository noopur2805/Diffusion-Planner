"""
NavSim preprocessing helper.

Generates two artefacts that ``NavSimMultiModalData`` consumes:

1) A camera manifest JSON: ``{token: {view: absolute_path}}`` for all 8 NavSim
   views in :data:`CAM_VIEWS`. Tokens are taken from the basenames of the
   existing vectorized ``.npz`` files (so this script does *not* require the
   NavSim devkit to be importable - paths are constructed by convention).

2) A data list JSON of the form ``[rel_path_1.npz, rel_path_2.npz, ...]``
   filtered to only those tokens that have all 8 camera images on disk.

Usage::

    python data_process_navsim.py \
        --vectorized_root /path/to/processed_navsim_npz \
        --camera_root /path/to/navsim_sensor_blobs \
        --out_manifest navsim_camera_manifest.json \
        --out_data_list navsim_train.json

The NavSim sensor blob layout (default after the official devkit download) is:
    <camera_root>/<log_name>/<token>/<view>.jpg
We discover that path automatically; users with a different layout can pass
``--layout flat`` (expects ``<camera_root>/<token>/<view>.jpg``).
"""
import argparse
import json
import os
from glob import glob

from diffusion_planner.model.module.camera_encoder import CameraEncoder

CAM_VIEWS = CameraEncoder.view_names()


def find_views_for_token(camera_root: str, token: str, layout: str):
    """Return {view: path} containing only existing files."""
    paths = {}
    if layout == "flat":
        token_dir = os.path.join(camera_root, token)
        for view in CAM_VIEWS:
            p = os.path.join(token_dir, f"{view}.jpg")
            if os.path.exists(p):
                paths[view] = p
        return paths

    # nested layout: <camera_root>/<log>/<token>/<view>.jpg
    matches = glob(os.path.join(camera_root, "*", token))
    for token_dir in matches:
        for view in CAM_VIEWS:
            p = os.path.join(token_dir, f"{view}.jpg")
            if os.path.exists(p):
                paths[view] = p
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorized_root", required=True, help="dir of preprocessed .npz scenes")
    parser.add_argument("--camera_root", required=True, help="root of NavSim camera images")
    parser.add_argument("--layout", choices=["flat", "nested"], default="nested")
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--out_data_list", required=True)
    parser.add_argument("--require_all_views", action="store_true", help="drop tokens missing any view")
    args = parser.parse_args()

    npz_files = sorted(
        os.path.relpath(p, args.vectorized_root)
        for p in glob(os.path.join(args.vectorized_root, "*.npz"))
    )
    print(f"Found {len(npz_files)} vectorized scenes under {args.vectorized_root}")

    manifest = {}
    kept = []
    for rel in npz_files:
        token = os.path.splitext(os.path.basename(rel))[0]
        views = find_views_for_token(args.camera_root, token, args.layout)
        if args.require_all_views and len(views) < len(CAM_VIEWS):
            continue
        if len(views) == 0:
            continue
        manifest[token] = views
        kept.append(rel)

    print(f"Wrote {len(manifest)} token->view entries; kept {len(kept)} scenes")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_manifest)) or ".", exist_ok=True)
    with open(args.out_manifest, "w") as fh:
        json.dump(manifest, fh)
    with open(args.out_data_list, "w") as fh:
        json.dump(kept, fh)


if __name__ == "__main__":
    main()
