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
import os
import sys
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

    def infer_video(self, input_path, output_video=None, handsonly_video=None,
                    wrist_json=None, hands_only=False,
                    rescale_factor=2.0, batch_size=8,
                    async_io=True, bg_color=(1.0, 1.0, 1.0)):
        """Run HaMeR on a single video.

        At least one of ``output_video``/``handsonly_video``/``wrist_json``
        must be given. If only ``wrist_json`` is given, no mp4 is written
        (fast path). If both ``output_video`` and ``handsonly_video`` are
        given, both videos are produced in a single inference pass.
        """
        if output_video is None and wrist_json is None and handsonly_video is None:
            raise ValueError("infer_video: need at least one output")
        if handsonly_video is not None and output_video is None:
            # produce only handsonly: use the legacy single-output path
            output_video, handsonly_video = handsonly_video, None
            hands_only = True
        wrist_only = output_video is None and handsonly_video is None
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
            handsonly_output_path=str(handsonly_video) if handsonly_video is not None else None,
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

        print(f"[hamer] {src.name} -> {full_out.name} + {hands_out.name}")
        process_video(
            str(src), str(full_out), pipe,
            rescale_factor=rescale_factor,
            batch_size=batch_size,
            bg_color=bg_color,
            async_io=async_io,
            wrist_json_path=str(wrist_out),
            detector_name=body_detector,
            handsonly_output_path=str(hands_out),
        )

        results.append({
            'input': str(src),
            'full': str(full_out),
            'handsonly': str(hands_out),
            'wrist_json': str(wrist_out),
        })

    return results


def _video_frame_count(path):
    """Return total frame count for ``path`` (0 if unreadable)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _lpt_split(items_with_weights, n_buckets):
    """Greedy Longest-Processing-Time scheduling.

    Args:
        items_with_weights: list of (item, weight) tuples.
        n_buckets: number of buckets.

    Returns:
        list of n_buckets lists (items only), with totals roughly balanced.
    """
    buckets = [[] for _ in range(n_buckets)]
    totals = [0] * n_buckets
    for item, w in sorted(items_with_weights, key=lambda x: -x[1]):
        i = min(range(n_buckets), key=lambda k: totals[k])
        buckets[i].append(item)
        totals[i] += w
    return buckets, totals


def hamer_parallel(
    input_dir,
    gpus,
    output_dir=None,
    extensions=('.mp4',),
    body_detector='regnety',
    batch_size=8,
    rescale_factor=2.0,
    async_io=True,
    recursive=False,
    skip_existing=True,
    wrist_only=False,
    python=None,
    log_dir=None,
):
    """Spawn one subprocess per GPU and split videos by frame count (LPT).

    Each worker is a fresh ``python hamer_api.py --files-from <list>`` with
    ``CUDA_VISIBLE_DEVICES=<gpu>`` so workers don't share GPU memory.

    Args:
        input_dir: Folder containing input videos.
        gpus: Iterable of GPU indices, e.g. ``[0, 1, 2, 3]``.
        log_dir: If given, each worker's stdout/stderr go to
            ``<log_dir>/worker_gpu<idx>.log``. Otherwise inherited.
        python: Python executable for workers (default: current interpreter).

    Returns:
        list of dicts: {'gpu', 'returncode', 'n_videos', 'log'} per worker.
    """
    import subprocess
    import tempfile

    gpus = list(gpus)
    if not gpus:
        raise ValueError("gpus must not be empty")
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exts_lower = tuple(e.lower() if e.startswith('.') else '.' + e.lower()
                       for e in extensions)
    walker = input_dir.rglob('*') if recursive else input_dir.iterdir()
    candidates = sorted(p for p in walker
                        if p.is_file() and p.suffix.lower() in exts_lower)
    if not candidates:
        print(f"[hamer-parallel] no videos with extensions {exts_lower} in {input_dir}")
        return []

    weights = [(p, _video_frame_count(p)) for p in candidates]
    buckets, totals = _lpt_split(weights, len(gpus))
    total_frames = sum(t for t in totals)
    print(f"[hamer-parallel] {len(candidates)} videos, {total_frames} frames, "
          f"split across {len(gpus)} GPUs:")
    for gpu, b, t in zip(gpus, buckets, totals):
        print(f"  GPU {gpu}: {len(b)} videos, {t} frames")

    python = python or sys.executable
    script = str(Path(__file__).resolve())
    list_dir = Path(tempfile.mkdtemp(prefix='hamer_parallel_'))
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for gpu, bucket in zip(gpus, buckets):
        if not bucket:
            continue
        list_path = list_dir / f'gpu{gpu}.txt'
        list_path.write_text('\n'.join(str(p) for p in bucket))
        cmd = [python, script, str(input_dir),
               '--output_dir', str(output_dir),
               '--files_from', str(list_path),
               '--body_detector', body_detector,
               '--batch_size', str(batch_size),
               '--rescale_factor', str(rescale_factor)]
        if not skip_existing:
            cmd.append('--no_skip_existing')
        if not async_io:
            cmd.append('--no_async_io')
        if wrist_only:
            cmd.append('--wrist_only')
        if recursive:
            cmd.append('--recursive')
        for e in extensions:
            cmd += ['--extensions', e]

        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if log_dir is not None:
            log_path = log_dir / f'worker_gpu{gpu}.log'
            log_f = open(log_path, 'w')
            proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        else:
            log_path = None
            log_f = None
            proc = subprocess.Popen(cmd, env=env)
        print(f"[hamer-parallel] spawned GPU {gpu} pid={proc.pid} "
              f"({len(bucket)} videos)" + (f" -> {log_path}" if log_path else ''))
        procs.append({'gpu': gpu, 'proc': proc, 'n_videos': len(bucket),
                      'log': str(log_path) if log_path else None, 'log_f': log_f})

    results = []
    for info in procs:
        rc = info['proc'].wait()
        if info['log_f'] is not None:
            info['log_f'].close()
        results.append({'gpu': info['gpu'], 'returncode': rc,
                        'n_videos': info['n_videos'], 'log': info['log']})
        print(f"[hamer-parallel] GPU {info['gpu']} exited rc={rc}")
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Batch HaMeR inference on a folder of videos')
    p.add_argument('input_dir')
    p.add_argument('--output_dir', default=None)
    p.add_argument('--extensions', nargs='+', default=['.mp4'])
    p.add_argument('--body_detector', default='regnety', choices=['vitdet', 'regnety'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--rescale_factor', type=float, default=2.0)
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--no_skip_existing', action='store_true')
    p.add_argument('--no_async_io', action='store_true')
    p.add_argument('--wrist_only', action='store_true',
                   help='Only write <stem>_wrist.json; skip mp4 output')
    p.add_argument('--files_from', default=None,
                   help='Path to a newline-separated list of video files. '
                        'Overrides directory walk; used by hamer_parallel workers.')
    p.add_argument('--gpus', default=None,
                   help='Comma-separated GPU indices, e.g. 0,1,2,3. '
                        'If given, dispatch via hamer_parallel.')
    p.add_argument('--log_dir', default=None,
                   help='Directory for per-worker logs (parallel mode only).')
    args = p.parse_args()

    if args.gpus is not None:
        gpu_list = [int(g.strip()) for g in args.gpus.split(',') if g.strip()]
        res = hamer_parallel(
            args.input_dir,
            gpus=gpu_list,
            output_dir=args.output_dir,
            extensions=tuple(args.extensions),
            body_detector=args.body_detector,
            batch_size=args.batch_size,
            rescale_factor=args.rescale_factor,
            recursive=args.recursive,
            skip_existing=not args.no_skip_existing,
            async_io=not args.no_async_io,
            wrist_only=args.wrist_only,
            log_dir=args.log_dir,
        )
        bad = [r for r in res if r['returncode'] != 0]
        print(f"[hamer-parallel] done. workers={len(res)} failed={len(bad)}")
        sys.exit(1 if bad else 0)

    files = None
    if args.files_from is not None:
        with open(args.files_from) as f:
            files = [line.strip() for line in f if line.strip()]

    out = hamer(
        args.input_dir,
        output_dir=args.output_dir,
        extensions=tuple(args.extensions),
        body_detector=args.body_detector,
        batch_size=args.batch_size,
        rescale_factor=args.rescale_factor,
        recursive=args.recursive,
        skip_existing=not args.no_skip_existing,
        async_io=not args.no_async_io,
        wrist_only=args.wrist_only,
        files=files,
    )
    print(f"[hamer] done. processed {len(out)} videos")
