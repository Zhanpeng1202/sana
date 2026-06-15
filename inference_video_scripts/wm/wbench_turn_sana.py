# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Sana-WM yaw turn-and-return generation over the WBench navigation split.

Mirrors the Matrix-Game-3 3D-consistency setup: for every WBench navigation
sample (its initial image + environment/character prompt) we drive Sana-WM with a
PURE-YAW "turn right to <angle> then back to the origin orientation" camera
trajectory and write one clean video per (case, angle). Generation only -- no
scoring.

Why explicit poses (not the --action DSL): the action DSL applies exponential
velocity smoothing and uses a fixed 0.6 deg/frame, so the realized angle is
approximate. We instead synthesize the exact camera-to-world trajectory with
Sana-WM's OWN ``rot_y`` (from camera_control), so "yaw right" matches Sana's
convention exactly and 75/180 deg are hit precisely -- and the trajectory is the
same pose convention (OpenCV c2w, relativized to frame 0) used by MG3 and lingbot,
so the three models are directly comparable.

Convention (verified against inference_sana_wm.py / camera_control.py):
  * --camera: (F,4,4) camera-to-world, OpenCV axes (+X right, +Y down, +Z fwd).
    Internally relativized to frame-0 identity, so only RELATIVE motion matters.
  * --intrinsics: [fx, fy, cx, cy] in the INPUT IMAGE's pixel coords (before the
    resize+center-crop to 704x1280); Sana transforms them for the crop. We build
    them per-image from a fixed horizontal FOV (default 60 deg) for cross-dataset
    consistency, analogous to MG3's fixed internal intrinsics.
  * num_frames must be 8k+1 (LTX-2 VAE constraint); we snap.

NOTE: this script does NOT run on import; call main(). Run later (GPU currently
occupied) via run_turn_wbench_sana.sh.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Repo root is on sys.path via the editable install (sana.egg-info); these import
# the heavy stack lazily inside main() to keep --help / py_compile cheap.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("sana_wm_wbench")


# --- WBench navigation sample enumeration (image + prompt per case) ---

def resolve_image(case: dict, data_dir: str) -> str | None:
    candidates = []
    img = case.get("settings", {}).get("initial_image", "")
    if img:
        candidates.append(img if os.path.isabs(img) else os.path.join(data_dir, img))
    candidates.append(os.path.join(data_dir, "images", f"case_{case['id']}.jpg"))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def select_cases(cases: list[dict], selection: str) -> list[dict]:
    def is_nav(i):
        return i.get("type") == "navigation"

    out = []
    for c in cases:
        inters = c.get("interactions", [])
        if not inters:
            continue
        if selection == "pure_nav" and all(is_nav(i) for i in inters):
            out.append(c)
        elif selection == "any_nav" and any(is_nav(i) for i in inters):
            out.append(c)
    return out


def build_prompt(case: dict) -> str:
    parts = [str(case.get("environment_prompt", "")).strip(),
             str(case.get("character_prompt", "")).strip()]
    return " ".join(p for p in parts if p)


def load_wbench_samples(wbench_root: str, data_dir: str | None, selection: str,
                        limit: int | None, case_ids: str | None) -> list[dict]:
    data_dir = data_dir or os.path.join(wbench_root, "data")
    files = sorted(glob.glob(os.path.join(data_dir, "cases", "case_*.json")),
                   key=lambda p: int(os.path.basename(p)[5:-5]))
    if not files:
        raise FileNotFoundError(f"no case_*.json under {data_dir}/cases")
    cases = [json.load(open(f)) for f in files]
    selected = select_cases(cases, selection)
    if case_ids:
        wanted = {x.strip() for x in case_ids.split(",") if x.strip()}
        selected = [c for c in selected if str(c["id"]) in wanted]
    if limit:
        selected = selected[:limit]
    samples = []
    for c in selected:
        image = resolve_image(c, data_dir)
        if image is None:
            logger.warning(f"case_{c['id']}: image not found, skipping")
            continue
        samples.append({"id": str(c["id"]), "image": image, "prompt": build_prompt(c)})
    return samples


# --- yaw turn-and-return trajectory (uses Sana's own rot_y for sign parity) ---

def build_yaw_turn_c2w(angle_deg: float, num_frames: int, direction: str, mode: str) -> np.ndarray:
    """(num_frames, 4, 4) camera-to-world pure-yaw trajectory, OpenCV axes.

    mode="return": ramp 0 -> angle over the first half, angle -> 0 over the second
                   (camera ends at the origin orientation -> loop-closure test).
    mode="turn":   one-way 0 -> angle across the whole clip.
    Translation is zero (rotate in place). Uses camera_control.rot_y so a positive
    angle is "yaw right" in Sana-WM's own convention.
    """
    from inference_video_scripts.wm.camera_control import rot_y

    sign = 1.0 if direction == "right" else -1.0
    if mode == "return":
        half = max(1, num_frames // 2)
        thetas = np.concatenate([
            np.linspace(0.0, angle_deg, half, endpoint=False),
            np.linspace(angle_deg, 0.0, num_frames - half),
        ])
    else:  # one-way
        thetas = np.linspace(0.0, angle_deg, num_frames)

    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (num_frames, 4, 4)).copy()
    for i, th in enumerate(thetas):
        poses[i, :3, :3] = rot_y(sign * math.radians(float(th))).astype(np.float32)
    return poses


def intrinsics_from_fov(width: int, height: int, fov_x_deg: float, num_frames: int) -> np.ndarray:
    """(num_frames, 4) [fx, fy, cx, cy] in the INPUT IMAGE's pixel coords."""
    fx = (width / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
    fy = fx  # square pixels
    cx, cy = width / 2.0, height / 2.0
    return np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (num_frames, 1))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sana-WM yaw turn-and-return over WBench navigation samples")
    # WBench selection
    ap.add_argument("--wbench_root", default="/home/builder/workspace/WBench")
    ap.add_argument("--data_dir", default=None, help="Defaults to <wbench_root>/data")
    ap.add_argument("--selection", choices=["pure_nav", "any_nav"], default="pure_nav")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--case_ids", default=None)
    ap.add_argument("--output_dir", default=None,
                    help="Defaults to <wbench_root>/work_dirs/sana_wm_turn/videos")
    ap.add_argument("--resume", action="store_true", default=False)

    # Trajectory
    ap.add_argument("--angles", default="75,180", help="Comma-separated yaw angles (deg).")
    ap.add_argument("--frames", type=int, default=201, help="Clip length (snapped to 8k+1).")
    ap.add_argument("--direction", choices=["right", "left"], default="right")
    ap.add_argument("--mode", choices=["return", "turn"], default="return",
                    help="return (default): turn out then back to origin (3D-consistency); turn: one-way.")
    ap.add_argument("--fov_deg", type=float, default=60.0, help="Assumed horizontal FOV for intrinsics.")
    ap.add_argument("--use_pi3x", action="store_true",
                    help="Estimate per-image intrinsics with Pi3X instead of a fixed FOV (slower).")

    # Generation knobs (mirror inference_sana_wm.py defaults)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--flow_shift", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--sampling_algo", default="flow_euler_ltx",
                    choices=["flow_euler_ltx", "flow_euler", "flow_dpm-solver", "self_forcing"])

    # Weights / config / refiner (defaults = public HF release)
    ap.add_argument("--config", default=None, help="Slim inference YAML (defaults to HF release).")
    ap.add_argument("--model_path", default=None, help="Stage-1 DiT checkpoint (defaults to HF release).")
    ap.add_argument("--no_refiner", action="store_true", help="Decode with Sana VAE instead of the LTX-2 refiner.")
    ap.add_argument("--refiner_root", default=None)
    ap.add_argument("--refiner_gemma_root", default=None)
    ap.add_argument("--refiner_seed", type=int, default=42)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--offload_vae", action="store_true")
    ap.add_argument("--offload_refiner", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Heavy imports (the editable-installed Sana stack). Done here so --help and
    # py_compile do not require torch / CUDA / model deps.
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    import torch

    from inference_video_scripts.wm.inference_sana_wm import (
        HF_DEFAULTS,
        GenerationParams,
        InferenceConfig,
        RefinerSettings,
        SanaWMPipeline,
        _snap_num_frames,
        estimate_intrinsics_with_pi3x,
        resize_and_center_crop,
        transform_intrinsics_for_crop,
        write_video,
    )
    import pyrallis
    from sana.tools import resolve_hf_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    angles = [float(a) for a in str(args.angles).split(",") if a.strip()]
    num_frames = _snap_num_frames(int(args.frames), stride=8)
    if num_frames != args.frames:
        logger.warning(f"LTX-2 VAE needs 8k+1 frames; --frames={args.frames} snapped to {num_frames}.")

    output_dir = Path(args.output_dir or os.path.join(args.wbench_root, "work_dirs", "sana_wm_turn", "videos"))
    samples = load_wbench_samples(args.wbench_root, args.data_dir, args.selection, args.limit, args.case_ids)
    logger.info(f"{len(samples)} sample(s) | angles={angles} dir={args.direction} mode={args.mode} "
                f"frames={num_frames} fov={args.fov_deg} refiner={not args.no_refiner}")

    # Resolve weights/config (HF release defaults).
    config_path = args.config or HF_DEFAULTS["config"]
    model_path = args.model_path or HF_DEFAULTS["model_path"]
    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=resolve_hf_path(config_path), args=[]
    )
    refiner = (
        None
        if args.no_refiner
        else RefinerSettings(
            root=args.refiner_root or HF_DEFAULTS["refiner_root"],
            gemma_root=args.refiner_gemma_root or HF_DEFAULTS["refiner_gemma_root"],
            sink_size=args.sink_size,
            seed=args.refiner_seed,
        )
    )

    # Build the trajectory ONCE per angle (identical across cases: pure yaw).
    trajectories = {a: build_yaw_turn_c2w(a, num_frames, args.direction, args.mode) for a in angles}

    # Load the model once.
    pipeline = SanaWMPipeline(
        config=config,
        model_path=resolve_hf_path(model_path),
        device=device,
        refiner=refiner,
        offload_vae=args.offload_vae,
        offload_refiner=args.offload_refiner,
        logger=logger,
    )

    n_ok, n_skip, n_fail = 0, 0, 0
    for s in samples:
        case_dir = output_dir / f"case_{s['id']}"
        for angle in angles:
            name = f"yaw_{args.direction}_{int(round(angle))}deg_{args.mode}_{num_frames}f"
            out_mp4 = case_dir / f"{name}_generated.mp4"
            if args.resume and out_mp4.exists():
                n_skip += 1
                logger.info(f"[case_{s['id']} {name}] SKIP (exists)")
                continue
            try:
                image = Image.open(s["image"]).convert("RGB")
                cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image)
                if args.use_pi3x:
                    intr_one = estimate_intrinsics_with_pi3x(image, device, logger)
                    intr_src = np.broadcast_to(intr_one, (num_frames, 4)).copy()
                else:
                    intr_src = intrinsics_from_fov(src_size[0], src_size[1], args.fov_deg, num_frames)
                intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)

                params = GenerationParams(
                    num_frames=num_frames, fps=args.fps, step=args.step, cfg_scale=args.cfg_scale,
                    flow_shift=args.flow_shift, seed=args.seed, negative_prompt=args.negative_prompt,
                    sampling_algo=args.sampling_algo,
                )
                out = pipeline.generate(cropped, s["prompt"], trajectories[angle], intrinsics_vec4, params)
                write_video(case_dir, name, out["video"], params.fps, logger)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                logger.error(f"[case_{s['id']} {name}] FAIL {e}")
                continue

    logger.info(f"Done - ok={n_ok} skip={n_skip} fail={n_fail} -> {output_dir}")


if __name__ == "__main__":
    main()
