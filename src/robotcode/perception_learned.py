"""Learned open-vocabulary object detector (OWLv2), a drop-in alternative to the colour detector.

    text queries from objects/catalog.yaml (every colour x type prompt, embedded once)
      -> OWLv2 on the RGB image -> 2D boxes, each with a score per query
      -> per box: depth pixels -> map-frame 3D points (camera kinematics + localisation)
         -> keep the object: near the box centre, in its depth layer, above its support surface
      -> the shared geometry (GeometricDetector): height, width, taper, support, centre from the top
      -> colour from the model's scores; type from metric size fused with the model's type scores

Nothing here is specific to saturated colours: any object the text encoder can name
can be proposed, and the depth geometry checks that it has the size of a catalog type.
On this simulator's primitive shapes the model's *type* label is weak (a red cylinder
scores higher as "cup" than as "bottle"), so metric size decides the type and the model
score only breaks near-ties (`type_weight`).

Optional dependency: pip install -e '.[vision]' (torch + transformers). The weights
(~600 MB) are downloaded from the Hugging Face hub on first use and cached in
~/.cache/huggingface. Runs on Apple-Silicon GPUs (MPS), CUDA or CPU.
"""
from __future__ import annotations

import math

import numpy as np

from .hal import RGBDFrame
from .perception import Detection, GeometricDetector

DEFAULTS = {
    "model": "google/owlv2-base-patch16-ensemble",
    "device": "auto",              # auto | mps | cuda | cpu
    "dtype": "auto",               # auto = float16 on mps/cuda (same accuracy, faster), float32 on cpu
    "image_size": 960,             # model input (square); 960 = the model's native resolution
    "template": "a photo of a {color} {type}",
    "score_threshold": 0.03,       # min. sigmoid score of a box for its best query
    "nms_iou": 0.3,
    "max_boxes": 30,
    "type_weight": 0.03,           # weight of the model's type log-odds vs. the metric size fit
    "object_radius": 0.10,         # 3D points kept around a box's core (m)
}

INSTALL_HINT = "the learned detector needs the optional vision dependencies: pip install -e '.[vision]'"


def require_vision() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"{INSTALL_HINT} ({exc})") from exc


def learned_config(catalog: dict) -> dict:
    return {**DEFAULTS, **catalog.get("perception", {}).get("learned", {})}


def build_queries(catalog: dict, template: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Every (colour, type) pair, once per prompt variant. Returns (texts, [(colour, type) per text])."""
    texts, keys = [], []
    for color, cspec in catalog["colors"].items():
        for cprompt in cspec.get("prompts", [color]):
            for otype, tspec in catalog["types"].items():
                for tprompt in tspec.get("prompts", [otype]):
                    texts.append(template.format(color=cprompt, type=tprompt))
                    keys.append((color, otype))
    return texts, keys


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy non-maximum suppression on (x0, y0, x1, y1) boxes; returns kept indices, best first."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = list(np.argsort(-scores))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        if not order:
            break
        rest = np.array(order)
        w = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        h = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        overlap = w * h / np.maximum(area[i] + area[rest] - w * h, 1e-9)
        order = [j for j, o in zip(order, overlap) if o <= iou]
    return keep


class OwlV2:
    """Image -> boxes (pixels, x0 y0 x1 y1) and per-query scores. Text embeddings are computed once."""

    def __init__(self, name: str, device: str = "auto", dtype: str = "auto", image_size: int = 960):
        require_vision()
        import torch
        from transformers import AutoTokenizer, Owlv2ForObjectDetection

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if dtype == "auto":
            dtype = "float32" if device == "cpu" else "float16"
        self.torch, self.device, self.dtype = torch, device, getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name).to(device=device, dtype=self.dtype).eval()
        self.size = int(image_size)
        self.native = self.model.config.vision_config.image_size
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        self.queries: list[str] = []
        self._query_embeds = None

    def set_queries(self, texts: list[str]) -> None:
        torch = self.torch
        tok = self.tokenizer(texts, padding="max_length", truncation=True, max_length=16, return_tensors="pt")
        with torch.inference_mode():
            out = self.model.owlv2.get_text_features(input_ids=tok["input_ids"].to(self.device),
                                                     attention_mask=tok["attention_mask"].to(self.device))
        emb = out if isinstance(out, torch.Tensor) else out.pooler_output
        self.queries, self._query_embeds = list(texts), emb[None]

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch, F = self.torch, self.torch.nn.functional
        h, w = rgb.shape[:2]
        side = max(h, w)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        x = F.pad(x, (0, side - w, 0, side - h))                       # pad bottom/right, as OWLv2 was trained
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype)
        interp = self.size != self.native
        m = self.model
        with torch.inference_mode():
            fmap, _ = m.image_embedder(pixel_values=x, interpolate_pos_encoding=interp)
            b, gh, gw, d = fmap.shape
            feats = fmap.reshape(b, gh * gw, d)
            logits, _ = m.class_predictor(feats, self._query_embeds.to(feats.dtype), None)
            boxes = m.box_predictor(feats, fmap, interp)
        scores = torch.sigmoid(logits[0].float()).cpu().numpy()
        cx, cy, bw, bh = (boxes[0].float().cpu().numpy() * side).T
        return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1), scores


_MODELS: dict[tuple, OwlV2] = {}


def load_model(cfg: dict) -> OwlV2:
    """One model per process (episodes in a benchmark worker reuse it)."""
    key = (cfg["model"], cfg["device"], cfg["dtype"], int(cfg["image_size"]))
    if key not in _MODELS:
        _MODELS[key] = OwlV2(*key)
    return _MODELS[key]


class LearnedObjectDetector(GeometricDetector):
    """OWLv2 boxes + depth geometry. Same detect() signature and Detection output as the colour detector."""

    kind = "learned"

    def __init__(self, ctx, model=None, **overrides):
        super().__init__(ctx)
        cfg = {**learned_config(ctx.catalog), **overrides}
        self.cfg = cfg
        self.texts, self.keys = build_queries(ctx.catalog, cfg["template"])
        self.colors = list(ctx.catalog["colors"])
        self.type_names = list(ctx.catalog["types"])
        # Column index sets: score of (colour, type) = best over its prompt variants.
        self.columns = {k: [i for i, kk in enumerate(self.keys) if kk == k] for k in set(self.keys)}
        self.model = model if model is not None else load_model(cfg)
        if getattr(self.model, "queries", None) != self.texts:
            self.model.set_queries(self.texts)
        self.last_boxes: list[dict] = []          # debug / benchmark: the 2D proposals of the last frame

    def _pair_scores(self, scores: np.ndarray) -> np.ndarray:
        """(N, Q) per-query scores -> (N, colours, types)."""
        out = np.zeros((scores.shape[0], len(self.colors), len(self.type_names)))
        for (c, t), cols in self.columns.items():
            out[:, self.colors.index(c), self.type_names.index(t)] = scores[:, cols].max(axis=1)
        return out

    def _detect(self, frame: RGBDFrame, pose, cam_pos, cam_rot) -> list[Detection]:
        boxes, scores = self.model(frame.rgb)
        best = scores.max(axis=1)
        cand = np.nonzero(best >= self.cfg["score_threshold"])[0]
        cand = cand[np.argsort(-best[cand])][: 4 * self.cfg["max_boxes"]]
        cand = cand[nms(boxes[cand], best[cand], self.cfg["nms_iou"])][: self.cfg["max_boxes"]]
        pairs = self._pair_scores(scores[cand])
        self.last_boxes = []
        detections: list[tuple[float, Detection]] = []
        for i, k in enumerate(cand):
            p = pairs[i]
            ci = int(np.argmax(p.max(axis=1)))
            color = self.colors[ci]
            box = {"box": boxes[k].round(1).tolist(), "score": float(best[k]), "color": color,
                   "model_type": self.type_names[int(np.argmax(p[ci]))]}
            self.last_boxes.append(box)
            blob = self._lift(boxes[k], frame, cam_pos, cam_rot, pose)
            if blob is None:
                box["rejected"] = "depth"
                continue
            fit = self.fit_errors(blob)
            # Model type evidence as a penalty: log-odds against the best type for this colour.
            ptype = np.maximum(p[ci], 1e-6)
            prior = {t: self.cfg["type_weight"] * min(math.log(ptype.max() / ptype[j]), 6.0)
                     for j, t in enumerate(self.type_names)}
            otype = min(fit, key=lambda t: fit[t] + prior[t])
            box.update(type=otype, fit_error=round(fit[otype], 3))
            if fit[otype] > self.MAX_FIT_ERROR:
                box["rejected"] = "geometry"
                continue
            detections.append((float(best[k]), self.make_detection(color, otype, fit[otype], blob, frame.stamp)))
        # Two boxes on one object (e.g. whole object and its top) -> keep the higher-scoring one.
        kept: list[Detection] = []
        for _, det in sorted(detections, key=lambda sd: -sd[0]):
            if all(np.linalg.norm(det.position[:2] - o.position[:2]) > 0.06 for o in kept):
                kept.append(det)
        return kept

    def _lift(self, box, frame: RGBDFrame, cam_pos, cam_rot, pose):
        """Box -> the object's own 3D points (drops background, support surface and neighbours)."""
        h, w = frame.depth.shape
        x0, y0 = max(int(math.floor(box[0])), 0), max(int(math.floor(box[1])), 0)
        x1, y1 = min(int(math.ceil(box[2])), w), min(int(math.ceil(box[3])), h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        sub = frame.depth[y0:y1, x0:x1]
        vs, us = np.nonzero((sub > 0) & (sub < self.max_range))
        if len(vs) < self.min_pixels:
            return None
        us, vs = us + x0, vs + y0
        d = frame.depth[vs, us]
        # The middle of the box is the object: its depth layer and footprint seed the selection.
        cu, cv = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        core = (np.abs(us + 0.5 - cu) <= 0.25 * (box[2] - box[0]) + 0.5) & \
               (np.abs(vs + 0.5 - cv) <= 0.25 * (box[3] - box[1]) + 0.5)
        if core.sum() < 3:
            return None
        keep = np.abs(d - np.median(d[core])) < 0.12
        us, vs, d, core = us[keep], vs[keep], d[keep], core[keep]
        xy, z = self.back_project(us, vs, d, frame, cam_pos, cam_rot, pose)
        seed = np.median(xy[core], axis=0)
        surface = self.ctx.prior.surface_at(seed[0], seed[1], margin=0.06)
        support_z = surface.top_z if surface else 0.0
        obj = (np.linalg.norm(xy - seed, axis=1) < self.cfg["object_radius"]) & (z > support_z + 0.01)
        if obj.sum() < self.min_pixels:
            return None
        return self.measure(xy[obj], z[obj], cam_pos, pose, surface)
