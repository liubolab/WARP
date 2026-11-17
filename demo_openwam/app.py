"""Minimal Gradio client for the official OpenWAM deployment server.

This file deliberately does not construct or replace the model.  Start the
official ``scripts/deploy.sh`` server with an OpenWAM checkpoint, then this UI
sends each image/video frame through the repository's WebSocket client and
renders the returned actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("OPENWAM_REPO", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _official_client():
    try:
        from benchmarks.utils.client import build_payload, encode_numpy_b64
        from benchmarks.utils.transport import WSPolicyClient
    except ImportError as exc:
        raise RuntimeError(
            "OpenWAM source tree/dependencies are missing. Run this app from the "
            "official repository and install its requirements (including websockets)."
        ) from exc
    return build_payload, encode_numpy_b64, WSPolicyClient


def _parse_state(text: str) -> list[list[float]] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"state JSON 无法解析: {exc.msg}") from exc
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [[float(x) for x in row] for row in value]
    if isinstance(value, list):
        return [[float(x) for x in value]]
    raise ValueError("state 必须是数字列表，或按帧排列的数字列表")


def _read_frames(path: str, max_frames: int) -> list[np.ndarray]:
    if not path:
        raise ValueError("请提供图像或视频")
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        from PIL import Image

        return [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)]

    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max(1, int(max_frames)):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError("视频没有可读取的帧")
    return frames


def predict(server_url: str, media_path: str, prompt: str, state_json: str, max_frames: int):
    build_payload, encode_numpy_b64, WSPolicyClient = _official_client()
    states = _parse_state(state_json)
    frames = _read_frames(media_path, max_frames)
    if states and len(states) not in (1, len(frames)):
        raise ValueError(f"state 提供了 {len(states)} 行；需要 1 行或与视频帧数一致的 {len(frames)} 行")
    actions: list[list[float]] = []
    try:
        with WSPolicyClient(server_url, timeout=600.0) as client:
            client.ping()
            for i, frame in enumerate(frames):
                # The official policy buffers action chunks. Reset per frame so
                # every displayed point comes from a fresh model forward using
                # that frame and its corresponding raw state.
                client.reset()
                state = None if states is None else states[0 if len(states) == 1 else i]
                payload = build_payload(
                    head=encode_numpy_b64(frame),
                    prompt=prompt or "",
                    state=state,
                )
                result = client.predict(payload)
                action = result.get("action")
                if not isinstance(action, list) or not action:
                    raise RuntimeError(f"官方 server 返回了无效 action: {result}")
                actions.append([float(x) for x in action])
    except Exception as exc:
        raise RuntimeError(f"真实 OpenWAM 推理失败（请确认 server/权重/状态维度）: {exc}") from exc

    width = max(len(row) for row in actions)
    table = pd.DataFrame(actions, columns=[f"action_{i}" for i in range(width)])
    table.insert(0, "frame", np.arange(len(table)))
    plot = table.set_index("frame").plot(kind="line", figsize=(10, 4), title="OpenWAM actions").figure
    return frames[0], table, plot, f"PASS: 收到 {len(actions)} 个真实 action；输入 {len(frames)} 帧。"


def build_demo(server_url: str):
    with gr.Blocks(title="OpenWAM minimal demo") as demo:
        with gr.Row():
            media = gr.File(label="真实图像/视频", file_types=["image", "video"], type="filepath")
            prompt = gr.Textbox(label="Prompt", value="")
        state = gr.Textbox(
            label="State JSON（必需时填 raw proprio；可填按帧二维列表）",
            placeholder='例如 [0.0, 0.0, ...] 或 [[...], [...]]',
            lines=2,
        )
        max_frames = gr.Slider(1, 64, value=16, step=1, label="视频最多发送帧数")
        run = gr.Button("运行真实推理", variant="primary")
        preview = gr.Image(label="输入预览")
        table = gr.Dataframe(label="动作数值表")
        plot = gr.Plot(label="动作预测曲线")
        status = gr.Textbox(label="状态", interactive=False)
        run.click(
            predict,
            [gr.State(server_url), media, prompt, state, max_frames],
            [preview, table, plot, status],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=os.environ.get("OPENWAM_SERVER_URL", "ws://127.0.0.1:8848"))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_demo(args.server).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
