"""External entry point for HaMeR inference.

Two layers:
    - ``Hamer`` class: build the pipeline once, then call ``infer_image`` /
      ``infer_video`` / ``infer_dir`` repeatedly. The recommended entry point.
    - ``hamer(input_dir, ...)`` function: folder-batch wrapper kept for
      backward compatibility (and used by the CLI). Internally delegates to
      ``Hamer.infer_dir``.

Usage (class):
    from hamer_api import Hamer
    h = Hamer(body_detector='regnety')
    out = h.infer_image('frame.jpg')                  # dict with 'hands'
    h.infer_video('clip.mp4', wrist_json='clip.json') # JSON only
    h.infer_dir('/path/to/videos', output_dir='out')

Usage (function, legacy):
    from hamer_api import hamer
    hamer('/path/to/videos', output_dir='/path/to/out')
"""
from pathlib import Path

import cv2
import numpy as np

from demo_video import build_pipeline, process_frame, process_video


def _load_image(image):
    """Accept BGR ndarray (uint8) or a file path; return BGR ndarray."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"could not read image: {image}")
        return img
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"image array must be uint8, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image array must be HxWx3 BGR, got shape {arr.shape}")
    return arr


class Hamer:
    """HaMeR pipeline held in memory. Build once, call many times."""

    def __init__(self, body_detector='regnety', checkpoint=None,
                 patch_renderer=True):
        kwargs = dict(body_detector=body_detector, patch_renderer=patch_renderer)
        if checkpoint is not None:
            kwargs['checkpoint'] = checkpoint
        self.pipe = build_pipeline(**kwargs)
        self.detector_name = body_detector

    def infer_image(self, image, render=False, hands_only=False,
                    rescale_factor=2.0, batch_size=8,
                    bg_color=(1.0, 1.0, 1.0)):
        """Run HaMeR on a single image.

        Args:
            image: BGR uint8 ndarray (HxWx3) or path-like to an image file.
            render: If True, also produce an overlay BGR image with meshes.
            hands_only: With render=True, render meshes on a flat ``bg_color``
                background instead of overlaying on the input.

        Returns:
            dict with keys:
                'hands' (list of per-hand dicts, same schema as JSON output),
                'width', 'height',
                'overlay' (BGR uint8 ndarray, only if render=True).
        """
        img = _load_image(image)
        overlay, _, hands = process_frame(
            img, self.pipe,
            rescale_factor=rescale_factor,
            batch_size=batch_size,
            hands_only=hands_only,
            bg_color=bg_color,
            return_hands_info=True,
            wrist_only=not render,
        )
        result = {'hands': hands, 'width': img.shape[1], 'height': img.shape[0]}
        if render:
            result['overlay'] = overlay
        return result

    def infer_video(self, input_path, output_video=None, wrist_json=None,
                    hands_only=False, rescale_factor=2.0, batch_size=8,
                    async_io=True, bg_color=(1.0, 1.0, 1.0)):
        """Run HaMeR on a single video.

        At least one of ``output_video`` or ``wrist_json`` must be given.
        If only ``wrist_json`` is given, no mp4 is written (fast path).
        """
        if output_video is None and wrist_json is None:
            raise ValueError("infer_video: need output_video or wrist_json")
        wrist_only = output_video is None
        # process_video requires output_path even in wrist_only mode (unused).
        out_path = str(output_video) if output_video is not None else '/dev/null'
        process_video(
            str(input_path), out_path, self.pipe,
            rescale_factor=rescale_factor,
            batch_size=batch_size,
            hands_only=hands_only,
            bg_color=bg_color,
            async_io=async_io,
            wrist_json_path=str(wrist_json) if wrist_json is not None else None,
            wrist_only=wrist_only,
            detector_name=self.detector_name,
        )

    def infer_dir(self, input_dir, **kwargs):
        """Folder-batch inference. Forwards to the legacy ``hamer`` function
        with this instance's pipeline."""
        return hamer(input_dir, pipe=self.pipe, body_detector=self.detector_name,
                     **kwargs)


def hamer(
    input_dir,
    output_dir=None,
    extensions=('.mp4',),
    body_detector='regnety',
    batch_size=8,
    rescale_factor=2.0,
    async_io=True,
    recursive=False,
    skip_existing=True,
    bg_color=(1.0, 1.0, 1.0),
    pipe=None,
    files=None,
    wrist_only=False,
):
    """Run HaMeR inference on every video in ``input_dir``.

    Args:
        input_dir: Folder containing input videos.
        output_dir: Where to write outputs. Defaults to ``input_dir``.
        extensions: Iterable of file extensions to match (case-insensitive).
        body_detector: 'vitdet' or 'regnety'.
        batch_size, rescale_factor, async_io: forwarded to process_video.
        recursive: If True, walk subdirectories too.
        skip_existing: If True, skip videos whose outputs both already exist.
        bg_color: RGB in [0,1] for the handsonly background.
        pipe: Pre-built pipeline dict from ``build_pipeline``. If None, one is
            built on first use and reused for all videos.

    Returns:
        List of dicts: [{'input', 'full', 'handsonly'}, ...] for processed
        videos (skipped ones are omitted).
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exts_lower = tuple(e.lower() if e.startswith('.') else '.' + e.lower()
                       for e in extensions)

    if files is not None:
        # Preserve caller-supplied order (e.g. for splitting work across workers).
        candidates = [Path(f) for f in files]
    elif recursive:
        candidates = sorted(p for p in input_dir.rglob('*')
                            if p.is_file() and p.suffix.lower() in exts_lower)
    else:
        candidates = sorted(p for p in input_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in exts_lower)

    if not candidates:
        print(f"[hamer] no videos with extensions {exts_lower} in {input_dir}")
        return []

    if pipe is None:
        pipe = build_pipeline(body_detector=body_detector)

    results = []
    for src in candidates:
        stem = src.stem
        full_out = output_dir / f"{stem}_full.mp4"
        hands_out = output_dir / f"{stem}_handsonly.mp4"
        wrist_out = output_dir / f"{stem}_wrist.json"

        if wrist_only:
            if skip_existing and wrist_out.exists():
                print(f"[hamer] skip (exists): {wrist_out.name}")
                continue
            print(f"[hamer] {src.name} -> {wrist_out.name} (wrist only)")
            process_video(
                str(src), str(full_out), pipe,
                rescale_factor=rescale_factor,
                batch_size=batch_size,
                async_io=async_io,
                wrist_json_path=str(wrist_out),
                wrist_only=True,
                detector_name=body_detector,
            )
            results.append({'input': str(src), 'wrist_json': str(wrist_out)})
            continue

        if (skip_existing and full_out.exists() and hands_out.exists()
                and wrist_out.exists()):
            print(f"[hamer] skip (exists): {src.name}")
            continue

        write_wrist = not (skip_existing and wrist_out.exists())
        print(f"[hamer] {src.name} -> {full_out.name}")
        if not (skip_existing and full_out.exists()) or write_wrist:
            process_video(
                str(src), str(full_out), pipe,
                rescale_factor=rescale_factor,
                batch_size=batch_size,
                hands_only=False,
                async_io=async_io,
                wrist_json_path=str(wrist_out) if write_wrist else None,
                detector_name=body_detector,
            )

        print(f"[hamer] {src.name} -> {hands_out.name}")
        if not (skip_existing and hands_out.exists()):
            process_video(
                str(src), str(hands_out), pipe,
                rescale_factor=rescale_factor,
                batch_size=batch_size,
                hands_only=True,
                bg_color=bg_color,
                async_io=async_io,
            )

        results.append({
            'input': str(src),
            'full': str(full_out),
            'handsonly': str(hands_out),
            'wrist_json': str(wrist_out),
        })

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Batch HaMeR inference on a folder of videos')
    p.add_argument('input_dir')
    p.add_argument('--output_dir', default=None)
    p.add_argument('--extensions', nargs='+', default=['.mp4'])
    p.add_argument('--body_detector', default='regnety', choices=['vitdet', 'regnety'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--no_skip_existing', action='store_true')
    p.add_argument('--no_async_io', action='store_true')
    p.add_argument('--wrist_only', action='store_true',
                   help='Only write <stem>_wrist.json; skip mp4 output')
    args = p.parse_args()

    out = hamer(
        args.input_dir,
        output_dir=args.output_dir,
        extensions=tuple(args.extensions),
        body_detector=args.body_detector,
        batch_size=args.batch_size,
        recursive=args.recursive,
        skip_existing=not args.no_skip_existing,
        async_io=not args.no_async_io,
        wrist_only=args.wrist_only,
    )
    print(f"[hamer] done. processed {len(out)} videos")
