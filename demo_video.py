"""Video demo for HaMeR.

Wraps the same pipeline as `demo.py` but takes a video file as input and writes
an overlay video as output. The official `demo.py` is left untouched; this
script reuses HaMeR/ViTPose/Detectron2 components as functions.

Usage:
    python demo_video.py --input path/to/in.mp4 --output path/to/out.mp4

Optional:
    --checkpoint, --batch_size, --rescale_factor, --body_detector,
    --max_frames, --stride, --side_view (saved as a separate side-view video)
"""
from pathlib import Path
import argparse
import os
import cv2
import numpy as np
import torch

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

from vitpose_model import ViTPoseModel

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


_HAMER_ROOT = Path(__file__).resolve().parent


def _patch_renderer_cache_offscreen():
    """Monkey-patch hamer.utils.renderer.Renderer.render_rgba_multiple to cache
    the pyrender.OffscreenRenderer instance keyed by (width, height) instead of
    creating/destroying it on every frame. EGL context churn was a major
    bottleneck; reusing the renderer yields ~30-50% speedup with identical
    output.
    """
    import pyrender
    from hamer.utils.renderer import Renderer, create_raymond_lights
    if getattr(Renderer, '_render_rgba_multiple_patched', False):
        return

    def render_rgba_multiple(
            self,
            vertices,
            cam_t,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
            **kwargs,  # accept side_view etc. silently if passed
        ):
        w = int(render_res[0]); h = int(render_res[1])
        cache = getattr(self, '_offscreen_cache', None)
        if cache is None:
            cache = {}
            self._offscreen_cache = cache
        key = (w, h)
        if key not in cache:
            cache[key] = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h, point_size=1.0)
        renderer = cache[key]

        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        mesh_list = [pyrender.Mesh.from_trimesh(
            self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle, is_right=sss)
        ) for vvv, ttt, sss in zip(vertices, cam_t, is_right)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3))
        for i, mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)
        camera_center = [w / 2., h / 2.]
        fl = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=fl, fy=fl, cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        for node in create_raymond_lights():
            scene.add_node(node)

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return color.astype(np.float32) / 255.0

    Renderer.render_rgba_multiple = render_rgba_multiple
    Renderer._render_rgba_multiple_patched = True


def build_pipeline(checkpoint=DEFAULT_CHECKPOINT, body_detector='vitdet', hamer_root=None,
                   patch_renderer=True):
    """Load HaMeR + body detector + ViTPose + Renderer. Returns a dict.

    The official HaMeR code references checkpoints via relative paths like
    ``./_DATA/...`` (e.g. MANO model path baked into the model config). This
    function temporarily chdir's to the HaMeR repo root during load and
    restores the caller's cwd before returning.
    """
    root = Path(hamer_root) if hamer_root else _HAMER_ROOT
    _orig_cwd = os.getcwd()
    os.chdir(root)
    if patch_renderer:
        _patch_renderer_cache_offscreen()
    try:
        download_models(CACHE_DIR_HAMER)
        model, model_cfg = load_hamer(checkpoint)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)
        model.eval()

        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        if body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = (
                "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            )
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif body_detector == 'regnety':
            from detectron2 import model_zoo
            detectron2_cfg = model_zoo.get_config(
                'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        else:
            raise ValueError(f"Unknown body_detector: {body_detector}")

        cpm = ViTPoseModel(device)
        renderer = Renderer(model_cfg, faces=model.mano.faces)
    finally:
        os.chdir(_orig_cwd)

    return {
        'model': model,
        'model_cfg': model_cfg,
        'detector': detector,
        'cpm': cpm,
        'renderer': renderer,
        'device': device,
    }


def detect_hand_bboxes(img_cv2, detector, cpm, min_bbox_side=64):
    """Run body detector + ViTPose, return (bboxes Nx4, is_right N) or (None, None).

    Per detected person ROI, ViTPose returns 21 left-hand and 21 right-hand
    keypoints. The original HaMeR demo emitted a bbox for both sides whenever
    each had >=4 confident points, which produced duplicate detections for a
    single visible hand (both the L and R keypoint sets fire for the same
    region). To prevent that we keep only the higher-confidence side per ROI
    (A in HAND_DEDUP_PLAN.md).

    A minimum bbox short-side filter rejects very small detections like
    monitor reflections (C in HAND_DEDUP_PLAN.md). Set ``min_bbox_side=0`` to
    disable.
    """
    det_out = detector(img_cv2)
    img_rgb = img_cv2[:, :, ::-1]

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    if len(pred_bboxes) == 0:
        return None, None

    vitposes_out = cpm.predict_pose(
        img_rgb,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes, is_right = [], []
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # A: pick the higher-confidence side per ROI.
        best = None  # (mean_conf, bbox, side)
        for keyp, side in [(left_hand_keyp, 0), (right_hand_keyp, 1)]:
            valid = keyp[:, 2] > 0.5
            if valid.sum() <= 3:
                continue
            mean_conf = float(keyp[valid, 2].mean())
            bbox = [float(keyp[valid, 0].min()), float(keyp[valid, 1].min()),
                    float(keyp[valid, 0].max()), float(keyp[valid, 1].max())]
            if best is None or mean_conf > best[0]:
                best = (mean_conf, bbox, side)
        if best is None:
            continue
        _, bbox, side = best

        # C: reject very small bboxes (likely reflections / false positives).
        if min_bbox_side > 0:
            short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if short_side < min_bbox_side:
                continue

        bboxes.append(bbox)
        is_right.append(side)

    if not bboxes:
        return None, None
    return np.stack(bboxes), np.stack(is_right)


def process_frame(img_cv2, pipe, rescale_factor=2.0, batch_size=8,
                  render_side=False, hands_only=False, bg_color=(1.0, 1.0, 1.0),
                  return_hands_info=False, wrist_only=False,
                  dual_output=False):
    """Run HaMeR on one BGR frame. Returns (main_bgr, side_bgr_or_None).

    If hands_only=True, the main output renders the hand meshes on a flat
    bg_color background (RGB in [0,1]) instead of overlaying on the input image.
    If no hands are detected, returns the input frame unchanged (overlay) or a
    blank background of the input size (hands_only).

    If dual_output=True, both composites are produced in a single inference
    pass and the function returns a 4-tuple
    ``(overlay_bgr, handsonly_bgr, side_bgr, hands_info)``. ``hands_only`` and
    ``return_hands_info`` are ignored in this mode (hands_info is always
    returned; both BGR images are always returned).

    If return_hands_info=True, returns (main_bgr, side_bgr_or_None, hands_info)
    where hands_info is a list of per-hand dicts with keys:
        'is_right' (int 0/1)
        'wrist_2d' [u, v] in pixels (full image)
        'bbox' [x1, y1, x2, y2] in full-image pixels (detector output)
        'joints_2d' (21, 2) MANO joints projected to full image pixels
        'joints_3d' (21, 3) MANO joints in camera frame (canonical x-mirror
                    applied so the values are visually consistent with the
                    image; same convention as vertices used for rendering)
        'cam_t' (3,) camera translation applied to the canonical hand
        'mano': {'global_orient': (1,3,3), 'hand_pose': (15,3,3), 'betas': (10,)}
                Raw HaMeR predictions in HaMeR's right-canonical frame; the
                x-mirror for left hands is NOT baked in here. Apply it
                downstream if you need a left-hand MANO mesh.
    """
    model = pipe['model']
    model_cfg = pipe['model_cfg']
    renderer = pipe['renderer']
    device = pipe['device']

    def _blank_like():
        bg = np.ones_like(img_cv2, dtype=np.float32)
        bg[:, :, 0] = bg_color[2] * 255  # B
        bg[:, :, 1] = bg_color[1] * 255  # G
        bg[:, :, 2] = bg_color[0] * 255  # R
        return bg.astype(np.uint8)

    bboxes, right = detect_hand_bboxes(img_cv2, pipe['detector'], pipe['cpm'])
    if bboxes is None:
        if dual_output:
            return img_cv2.copy(), _blank_like(), (_blank_like() if render_side else None), []
        if wrist_only:
            return None, None, []
        main = _blank_like() if hands_only else img_cv2.copy()
        side = (_blank_like() if hands_only else img_cv2.copy()) if render_side else None
        if return_hands_info:
            return main, side, []
        return main, side

    dataset = ViTDetDataset(model_cfg, img_cv2, bboxes, right, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_verts, all_cam_t, all_right = [], [], []
    hands_info = []
    scaled_focal_length = None
    img_size_last = None
    bbox_idx = 0

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size, scaled_focal_length
        ).detach().cpu().numpy()

        joints_np = out['pred_keypoints_3d'].detach().cpu().numpy()
        need_verts = dual_output or (not wrist_only)
        verts_np = out['pred_vertices'].detach().cpu().numpy() if need_verts else None
        mano_go = out['pred_mano_params']['global_orient'].detach().cpu().numpy()
        mano_hp = out['pred_mano_params']['hand_pose'].detach().cpu().numpy()
        mano_bt = out['pred_mano_params']['betas'].detach().cpu().numpy()

        for n in range(batch['img'].shape[0]):
            is_right_n = batch['right'][n].cpu().numpy()
            sign = 2 * float(is_right_n) - 1
            if need_verts:
                verts = verts_np[n]
                verts[:, 0] = sign * verts[:, 0]
                all_verts.append(verts)
                all_cam_t.append(pred_cam_t_full[n])
                all_right.append(is_right_n)
            img_size_last = img_size[n]

            iw = float(img_size[n, 0].item())
            ih = float(img_size[n, 1].item())
            fl = float(scaled_focal_length.item()) if hasattr(scaled_focal_length, 'item') else float(scaled_focal_length)
            cam_t = pred_cam_t_full[n]

            joints_local = joints_np[n].copy()  # (21, 3)
            joints_local[:, 0] = sign * joints_local[:, 0]
            joints_cam = joints_local + cam_t[None, :]
            Z = joints_cam[:, 2]
            safe_Z = np.where(Z > 1e-6, Z, 1.0)
            u_all = fl * joints_cam[:, 0] / safe_Z + iw / 2.0
            v_all = fl * joints_cam[:, 1] / safe_Z + ih / 2.0
            joints_2d = np.stack([u_all, v_all], axis=-1)
            joints_2d[Z <= 1e-6] = float('nan')

            bbox_xyxy = bboxes[bbox_idx].tolist()
            bbox_idx += 1

            hands_info.append({
                'is_right': int(is_right_n),
                'wrist_2d': [float(joints_2d[0, 0]), float(joints_2d[0, 1])],
                'bbox': [float(v) for v in bbox_xyxy],
                'joints_2d': joints_2d.tolist(),
                'joints_3d': joints_cam.tolist(),
                'cam_t': [float(cam_t[0]), float(cam_t[1]), float(cam_t[2])],
                'mano': {
                    'global_orient': mano_go[n].tolist(),
                    'hand_pose': mano_hp[n].tolist(),
                    'betas': mano_bt[n].tolist(),
                },
            })

    if wrist_only and not dual_output:
        return None, None, hands_info

    if not all_verts:
        if dual_output:
            return img_cv2.copy(), _blank_like(), (_blank_like() if render_side else None), hands_info
        main = _blank_like() if hands_only else img_cv2.copy()
        side = (_blank_like() if hands_only else img_cv2.copy()) if render_side else None
        if return_hands_info:
            return main, side, hands_info
        return main, side

    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )
    cam_view = renderer.render_rgba_multiple(
        all_verts, cam_t=all_cam_t, render_res=img_size_last,
        is_right=all_right, **misc_args
    )

    input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
    alpha = cam_view[:, :, 3:]
    rgb = cam_view[:, :, :3]

    def _to_bgr(comp):
        return (255 * comp[:, :, ::-1]).clip(0, 255).astype(np.uint8)

    bg_flat = np.empty_like(input_img)
    bg_flat[:, :, 0] = bg_color[0]
    bg_flat[:, :, 1] = bg_color[1]
    bg_flat[:, :, 2] = bg_color[2]

    overlay_bgr = handsonly_bgr = None
    if dual_output:
        overlay_bgr = _to_bgr(input_img * (1 - alpha) + rgb * alpha)
        handsonly_bgr = _to_bgr(bg_flat * (1 - alpha) + rgb * alpha)
    else:
        bg_img = bg_flat if hands_only else input_img
        overlay_bgr = _to_bgr(bg_img * (1 - alpha) + rgb * alpha)

    side_bgr = None
    if render_side:
        side_view = renderer.render_rgba_multiple(
            all_verts, cam_t=all_cam_t, render_res=img_size_last,
            is_right=all_right, side_view=True, **misc_args
        )
        white = np.ones_like(input_img)
        side_blend = white * (1 - side_view[:, :, 3:]) + side_view[:, :, :3] * side_view[:, :, 3:]
        side_bgr = _to_bgr(side_blend)

    if dual_output:
        return overlay_bgr, handsonly_bgr, side_bgr, hands_info
    if return_hands_info:
        return overlay_bgr, side_bgr, hands_info
    return overlay_bgr, side_bgr


def process_video(input_path, output_path, pipe,
                  rescale_factor=2.0, batch_size=8,
                  max_frames=None, stride=1,
                  start_frame=0, end_frame=None,
                  side_output_path=None,
                  hands_only=False, bg_color=(1.0, 1.0, 1.0),
                  async_io=False, prefetch=4,
                  wrist_json_path=None, wrist_only=False,
                  detector_name=None,
                  handsonly_output_path=None):
    """Run HaMeR on a video.

    If ``handsonly_output_path`` is set alongside ``output_path``, both an
    overlay video and a handsonly video are written in a **single inference
    pass** (the meshes are rendered once per frame and composited twice). In
    that case the ``hands_only`` argument is ignored.
    """
    if wrist_only and wrist_json_path is None:
        raise ValueError("wrist_only=True requires wrist_json_path")
    if wrist_only and handsonly_output_path is not None:
        raise ValueError("wrist_only=True is incompatible with handsonly_output_path")
    dual = handsonly_output_path is not None and not wrist_only
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    handsonly_writer = None
    side_writer = None
    if not wrist_only:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps / max(stride, 1), (width, height))
        if dual:
            os.makedirs(os.path.dirname(os.path.abspath(handsonly_output_path)) or '.', exist_ok=True)
            handsonly_writer = cv2.VideoWriter(str(handsonly_output_path), fourcc, fps / max(stride, 1), (width, height))
        if side_output_path is not None:
            side_writer = cv2.VideoWriter(str(side_output_path), fourcc, fps / max(stride, 1), (width, height))

    want_wrist = wrist_json_path is not None
    wrist_records = [] if want_wrist else None
    model_cfg = pipe['model_cfg']
    focal_length_px = float(
        model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(width, height)
    )

    if async_io:
        import threading, queue as _queue
        in_q = _queue.Queue(maxsize=prefetch)
        out_q = _queue.Queue(maxsize=prefetch)
        SENTINEL = object()

        def _decoder():
            fi = start_frame
            while fi < end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                if (fi - start_frame) % stride == 0:
                    in_q.put((fi, frame))
                fi += 1
            in_q.put(SENTINEL)

        def _writer_th():
            while True:
                item = out_q.get()
                if item is SENTINEL:
                    break
                ov, ho, sd = item
                if writer is not None and ov is not None:
                    writer.write(ov)
                if handsonly_writer is not None and ho is not None:
                    handsonly_writer.write(ho)
                if side_writer is not None and sd is not None:
                    side_writer.write(sd)

        dec_th = threading.Thread(target=_decoder, daemon=True)
        wr_th = threading.Thread(target=_writer_th, daemon=True)
        dec_th.start(); wr_th.start()

        written = 0
        try:
            while True:
                item = in_q.get()
                if item is SENTINEL:
                    break
                fi, frame = item
                if dual:
                    overlay, handsonly, side, hinfo = process_frame(
                        frame, pipe,
                        rescale_factor=rescale_factor,
                        batch_size=batch_size,
                        render_side=(side_writer is not None),
                        bg_color=bg_color,
                        dual_output=True,
                    )
                else:
                    overlay, side, hinfo = process_frame(
                        frame, pipe,
                        rescale_factor=rescale_factor,
                        batch_size=batch_size,
                        render_side=(side_writer is not None),
                        hands_only=hands_only,
                        bg_color=bg_color,
                        return_hands_info=True,
                        wrist_only=wrist_only,
                    )
                    handsonly = None
                if want_wrist:
                    wrist_records.append({'frame': fi, 'hands': hinfo})
                out_q.put((overlay, handsonly, side))
                written += 1
                if written % 10 == 0:
                    print(f"[{written}] frame {fi}/{end_frame} (range {start_frame}-{end_frame}) [async]", flush=True)
                if max_frames is not None and written >= max_frames:
                    break
        finally:
            out_q.put(SENTINEL)
            dec_th.join(timeout=5)
            wr_th.join(timeout=10)
            cap.release()
            if writer is not None:
                writer.release()
            if handsonly_writer is not None:
                handsonly_writer.release()
            if side_writer is not None:
                side_writer.release()
        if want_wrist:
            _write_wrist_json(wrist_json_path, wrist_records, input_path, width, height, fps, focal_length=focal_length_px, detector=detector_name)
        if not wrist_only:
            print(f"Done. Wrote {written} frames to {output_path}", flush=True)
        return

    frame_idx = start_frame
    written = 0
    try:
        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            local_idx = frame_idx - start_frame
            if local_idx % stride == 0:
                if dual:
                    overlay, handsonly, side, hinfo = process_frame(
                        frame, pipe,
                        rescale_factor=rescale_factor,
                        batch_size=batch_size,
                        render_side=(side_writer is not None),
                        bg_color=bg_color,
                        dual_output=True,
                    )
                else:
                    overlay, side, hinfo = process_frame(
                        frame, pipe,
                        rescale_factor=rescale_factor,
                        batch_size=batch_size,
                        render_side=(side_writer is not None),
                        hands_only=hands_only,
                        bg_color=bg_color,
                        return_hands_info=True,
                        wrist_only=wrist_only,
                    )
                    handsonly = None
                if want_wrist:
                    wrist_records.append({'frame': frame_idx, 'hands': hinfo})
                if writer is not None and overlay is not None:
                    writer.write(overlay)
                if handsonly_writer is not None and handsonly is not None:
                    handsonly_writer.write(handsonly)
                if side_writer is not None and side is not None:
                    side_writer.write(side)
                written += 1
                if written % 10 == 0:
                    print(f"[{written}] frame {frame_idx}/{end_frame} (range {start_frame}-{end_frame})", flush=True)
                if max_frames is not None and written >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if handsonly_writer is not None:
            handsonly_writer.release()
        if side_writer is not None:
            side_writer.release()

    if want_wrist:
        _write_wrist_json(wrist_json_path, wrist_records, input_path, width, height, fps, focal_length=focal_length_px, detector=detector_name)
    if not wrist_only:
        print(f"Done. Wrote {written} frames to {output_path}", flush=True)


def _write_wrist_json(path, records, input_path, width, height, fps,
                      focal_length=None, detector=None):
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    payload = {
        'schema_version': 2,
        'input': str(input_path),
        'width': int(width),
        'height': int(height),
        'fps': float(fps),
        'n_frames_processed': len(records),
        'detector': detector,
        'camera': {
            'focal_length_px': float(focal_length) if focal_length is not None else None,
            'principal_point_px': [float(width) / 2.0, float(height) / 2.0],
            'model': 'pinhole, origin top-left, +x right, +y down, +z forward',
        },
        'units': {
            'wrist_2d': 'pixels (full image)',
            'joints_2d': 'pixels (full image), shape (21, 2)',
            'joints_3d': 'camera frame, MANO units (~meters), shape (21, 3); x-mirror applied for left hands',
            'cam_t': 'camera frame translation applied to canonical hand, shape (3,)',
            'mano': "raw HaMeR predictions in HaMeR's right-canonical frame; "
                    "global_orient (1,3,3), hand_pose (15,3,3), betas (10,). "
                    "x-mirror for left hands is NOT baked in.",
        },
        'frames': records,
    }
    with open(path, 'w') as f:
        json.dump(payload, f)
    print(f"Done. Wrote wrist JSON to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='HaMeR video demo')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output video path (.mp4)')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--start_frame', type=int, default=0, help='Frame index to start from (inclusive)')
    parser.add_argument('--end_frame', type=int, default=None, help='Frame index to stop at (exclusive)')
    parser.add_argument('--side_view_output', type=str, default=None,
                        help='Optional path to also save a side-view video')
    parser.add_argument('--hands_only', action='store_true',
                        help='Render hand meshes on a flat background instead of overlaying on input')
    parser.add_argument('--bg_color', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        metavar=('R', 'G', 'B'), help='Background color for --hands_only (0-1 range)')
    parser.add_argument('--async_io', action='store_true',
                        help='Run video decode and output encode on background threads')
    parser.add_argument('--prefetch', type=int, default=4,
                        help='Queue size for async decode/write')
    args = parser.parse_args()

    pipe = build_pipeline(checkpoint=args.checkpoint, body_detector=args.body_detector)
    process_video(
        args.input, args.output, pipe,
        rescale_factor=args.rescale_factor,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        stride=args.stride,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        side_output_path=args.side_view_output,
        hands_only=args.hands_only,
        bg_color=tuple(args.bg_color),
        async_io=args.async_io,
        prefetch=args.prefetch,
    )


if __name__ == '__main__':
    main()
