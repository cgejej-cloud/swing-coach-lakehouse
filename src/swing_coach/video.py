"""Video decoding through ffmpeg (OpenCV cannot open non-ASCII Windows paths)."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int


def _ffprobe() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    return str(Path(ffmpeg).with_name("ffprobe" + Path(ffmpeg).suffix))


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        [_ffprobe(), "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
         "-of", "json", str(path)],
        capture_output=True, check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return VideoInfo(int(s["width"]), int(s["height"]), int(num) / int(den), int(s["nb_read_packets"]))


def frames(path: Path, info: VideoInfo) -> Iterator[np.ndarray]:
    """Yield RGB frames (H, W, 3) uint8."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # stopping early closes the pipe; ffmpeg then reports a broken pipe
    )
    size = info.width * info.height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
    finally:
        proc.stdout.close()
        proc.wait()
