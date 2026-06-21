from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_VIDEO = r"c:\Users\gabri\Downloads\brasilxmorroco.mp4"
DEFAULT_MODEL = str(Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx")
DEFAULT_EMOTION_MODEL = str(Path(__file__).resolve().parent / "models" / "emotion-ferplus-8.onnx")
FERPLUS_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

# Peso de positividade de cada emocao (0 = nada confiante, 1 = totalmente confiante).
# Media ponderada das probabilidades; "neutral" fica no meio para gerar variacao real.
EMOTION_WEIGHTS = {
    "happiness": 1.00,
    "neutral": 0.60,
    "surprise": 0.50,
    "contempt": 0.35,
    "sadness": 0.20,
    "anger": 0.15,
    "disgust": 0.15,
    "fear": 0.10,
}


@dataclass
class EmotionResult:
    confidence: float
    reliability: float
    dominant: str
    emotion_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    result: EmotionResult


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    missed: int = 0
    confidence_values: list[float] = field(default_factory=list)
    reliability_values: list[float] = field(default_factory=list)
    dominant_values: list[str] = field(default_factory=list)

    def update(self, detection: Detection) -> None:
        if self.confidence_values:
            self.bbox = smooth_bbox(self.bbox, detection.bbox)
        else:
            self.bbox = detection.bbox
        self.missed = 0
        self.confidence_values.append(detection.result.confidence)
        self.reliability_values.append(detection.result.reliability)
        self.dominant_values.append(detection.result.dominant)

    @property
    def avg_confidence(self) -> float:
        return statistics.fmean(self.confidence_values) if self.confidence_values else 0.0

    @property
    def avg_reliability(self) -> float:
        return statistics.fmean(self.reliability_values) if self.reliability_values else 0.0

    @property
    def dominant_label(self) -> str:
        if not self.dominant_values:
            return "unknown"
        return max(set(self.dominant_values), key=self.dominant_values.count)


class CentroidTracker:
    def __init__(self, max_distance: float = 90.0, max_missed: int = 45) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> dict[int, Track]:
        for track in self.tracks.values():
            track.missed += 1

        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())
        pairs: list[tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            tx, ty = center(track.bbox)
            for det_idx, detection in enumerate(detections):
                dx, dy = center(detection.bbox)
                distance = math.hypot(tx - dx, ty - dy)
                if distance <= self.max_distance:
                    pairs.append((distance, track_id, det_idx))

        for _, track_id, det_idx in sorted(pairs, key=lambda item: item[0]):
            if track_id not in unmatched_tracks or det_idx not in unmatched_detections:
                continue
            self.tracks[track_id].update(detections[det_idx])
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(det_idx)

        for det_idx in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            track = Track(track_id=track_id, bbox=detections[det_idx].bbox)
            track.update(detections[det_idx])
            self.tracks[track_id] = track

        for track_id in list(unmatched_tracks):
            if self.tracks[track_id].missed > self.max_missed:
                del self.tracks[track_id]

        return self.tracks


class ExpressionAnalyzer:
    def __init__(
        self,
        min_face_frac: float = 0.08,
        min_skin_frac: float = 0.30,
        model_path: str = DEFAULT_MODEL,
        emotion_model_path: str = DEFAULT_EMOTION_MODEL,
        score_threshold: float = 0.85,
        detect_scale: float = 0.6,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo YuNet nao encontrado em {model_path}.")
        if not Path(emotion_model_path).exists():
            raise FileNotFoundError(f"Modelo FER+ nao encontrado em {emotion_model_path}.")

        self.min_face_frac = min_face_frac
        self.min_skin_frac = min_skin_frac
        self.score_threshold = score_threshold
        self.detect_scale = detect_scale
        self.face_detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        self._input_size: tuple[int, int] | None = None
        self.emotion_net = cv2.dnn.readNetFromONNX(emotion_model_path)

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        scale = self.detect_scale
        resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape[:2]
        if self._input_size != (rw, rh):
            self.face_detector.setInputSize((rw, rh))
            self._input_size = (rw, rh)

        _, faces = self.face_detector.detect(resized)
        if faces is None:
            return []

        min_px = self.min_face_frac * frame.shape[0]
        inv = 1.0 / scale
        boxes: list[tuple[int, int, int, int]] = []
        for face in faces:
            score = float(face[-1])
            if score < self.score_threshold:
                continue
            bx, by = int(face[0] * inv), int(face[1] * inv)
            bw, bh = int(face[2] * inv), int(face[3] * inv)
            if bh < min_px:
                continue
            crop = frame[max(0, by) : by + bh, max(0, bx) : bx + bw]
            if skin_fraction(crop) < self.min_skin_frac:
                continue
            boxes.append((bx, by, bw, bh))
        return non_max_suppression(boxes, overlap_threshold=0.30)

    def analyze(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> EmotionResult:
        x, y, w, h = clamp_bbox(bbox, frame.shape[1], frame.shape[0])
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return EmotionResult(0.0, 0.0, "unknown")

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (64, 64)).astype(np.float32)
            self.emotion_net.setInput(face.reshape(1, 1, 64, 64))
            logits = self.emotion_net.forward().flatten()
        except cv2.error:
            return EmotionResult(0.0, 0.0, "unknown")

        exps = np.exp(logits - float(np.max(logits)))
        probs = exps / float(np.sum(exps))
        scores = {FERPLUS_LABELS[i]: float(probs[i]) for i in range(len(FERPLUS_LABELS))}

        confidence = 100.0 * sum(EMOTION_WEIGHTS[name] * prob for name, prob in scores.items())
        return EmotionResult(
            confidence=float(np.clip(confidence, 0.0, 100.0)),
            reliability=crop_quality(crop),
            dominant=max(scores, key=scores.get),
            emotion_scores=scores,
        )


def detection_rank(detection: Detection) -> float:
    _, _, w, h = detection.bbox
    return float(w * h) * (0.5 + 0.5 * detection.result.reliability)


def center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def smooth_bbox(
    previous: tuple[int, int, int, int],
    current: tuple[int, int, int, int],
    alpha: float = 0.65,
) -> tuple[int, int, int, int]:
    px, py, pw, ph = previous
    cx, cy, cw, ch = current
    return (
        int(px * alpha + cx * (1.0 - alpha)),
        int(py * alpha + cy * (1.0 - alpha)),
        int(pw * alpha + cw * (1.0 - alpha)),
        int(ph * alpha + ch * (1.0 - alpha)),
    )


def clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def skin_fraction(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return float(np.count_nonzero(mask)) / float(mask.size)


def crop_quality(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    area_score = min((w * h) / (120 * 120), 1.0)
    sharpness_score = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 120.0, 1.0)
    brightness = float(np.mean(gray)) / 255.0
    brightness_score = 1.0 - min(abs(brightness - 0.5) / 0.5, 1.0)
    return float(np.clip(area_score * 0.45 + sharpness_score * 0.35 + brightness_score * 0.20, 0.0, 1.0))


def non_max_suppression(
    boxes: list[tuple[int, int, int, int]], overlap_threshold: float
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    arr = np.array(boxes, dtype=float)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 0] + arr[:, 2]
    y2 = arr[:, 1] + arr[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    picked: list[int] = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        picked.append(int(i))
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    return [boxes[i] for i in picked]


def draw_track(frame: np.ndarray, track: Track) -> None:
    x, y, w, h = clamp_bbox(face_only_bbox(track.bbox), frame.shape[1], frame.shape[0])
    label, color = score_label_and_color(track.avg_confidence)
    text = f"{label} {track.avg_confidence:.0f}%"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    draw_label(frame, text, x, y, w, color)


def face_only_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    new_w = int(w * 0.78)
    new_h = int(h * 0.82)
    new_x = x + int((w - new_w) / 2)
    new_y = y + int(h * 0.06)
    return new_x, new_y, new_w, new_h


def draw_label(
    frame: np.ndarray,
    text: str,
    box_x: int,
    box_y: int,
    box_w: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(np.clip(box_w / 150.0, 0.55, 1.1))
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    bg_w = tw + pad * 2
    bg_h = th + baseline + pad * 2

    x = box_x
    y = box_y - bg_h
    if y < 0:
        y = box_y
    x = max(0, min(x, frame.shape[1] - bg_w))

    cv2.rectangle(frame, (x, y), (x + bg_w, y + bg_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + bg_w, y + bg_h), color, 1)
    text_org = (x + pad, y + pad + th)
    cv2.putText(frame, text, text_org, font, scale, color, thickness, cv2.LINE_AA)


def score_label_and_color(score: float) -> tuple[str, tuple[int, int, int]]:
    # Neutro/calmo (~60) fica Confiante; so cai para Intimidado quando ha tensao real
    # puxando o indice para baixo, evitando rotular intimidado por incerteza.
    if score >= 50:
        return "Confiante", (40, 180, 40)
    if score >= 30:
        return "Intimidado", (0, 255, 255)
    return "Desacreditado", (45, 45, 220)


def write_summary(csv_path: Path, tracks: list[Track], min_samples: int) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "jogador_id",
                "confianca_visual_media_percent",
                "confiabilidade_media_percent",
                "amostras",
                "expressao_dominante",
            ]
        )
        valid_tracks = [t for t in tracks if len(t.confidence_values) >= min_samples]
        for track in sorted(valid_tracks, key=lambda item: item.avg_confidence, reverse=True):
            writer.writerow(
                [
                    track.track_id,
                    round(track.avg_confidence, 2),
                    round(track.avg_reliability * 100.0, 2),
                    len(track.confidence_values),
                    track.dominant_label,
                ]
            )


def analyze_video(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Video nao encontrado: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / args.output_video
    output_csv = output_dir / args.output_csv

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    analyzer = ExpressionAnalyzer(
        min_face_frac=args.min_face_frac,
        min_skin_frac=args.min_skin_frac,
        model_path=args.model,
        emotion_model_path=args.emotion_model,
        score_threshold=args.score_threshold,
        detect_scale=args.detect_scale,
    )
    tracker = CentroidTracker(max_distance=args.max_track_distance, max_missed=args.max_missed)
    all_tracks: dict[int, Track] = {}
    frame_index = 0

    with tqdm(total=frame_count or None, desc="Analisando video", unit="frame") as progress:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections: list[Detection] = []
            if frame_index % args.detect_every == 0:
                candidates: list[Detection] = []
                for bbox in analyzer.detect_faces(frame):
                    result = analyzer.analyze(frame, bbox)
                    if result.reliability >= args.min_reliability:
                        candidates.append(Detection(bbox=bbox, result=result))
                candidates.sort(key=detection_rank, reverse=True)
                detections = candidates[: args.max_faces]

            active_tracks = tracker.update(detections)
            for track in active_tracks.values():
                all_tracks[track.track_id] = track

            drawable = [
                t
                for t in active_tracks.values()
                if t.missed <= args.max_draw_missed and len(t.confidence_values) >= args.min_draw_hits
            ]
            drawable.sort(key=lambda t: t.bbox[2] * t.bbox[3], reverse=True)
            for track in drawable[: args.max_faces]:
                draw_track(frame, track)

            writer.write(frame)
            frame_index += 1
            progress.update(1)
            if args.max_frames and frame_index >= args.max_frames:
                break

    capture.release()
    writer.release()
    write_summary(output_csv, list(all_tracks.values()), min_samples=args.min_summary_samples)
    print(f"Video anotado: {output_video}")
    print(f"Resumo CSV: {output_csv}")
    print("Observacao: a pontuacao e um indice visual aproximado, nao uma leitura real do estado mental.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta rostos de jogadores em um video e estima um indice visual de confianca."
    )
    parser.add_argument("--input", default=DEFAULT_VIDEO, help="Caminho do video de entrada.")
    parser.add_argument("--output-dir", default="outputs", help="Pasta de saida.")
    parser.add_argument("--output-video", default=f"annotated_{int(time.time())}.mp4", help="Nome do video anotado.")
    parser.add_argument("--output-csv", default="confidence_summary.csv", help="Nome do CSV de resumo.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Caminho do modelo YuNet .onnx.")
    parser.add_argument("--emotion-model", default=DEFAULT_EMOTION_MODEL, help="Caminho do modelo de emocao FER+ .onnx.")
    parser.add_argument("--detect-every", type=int, default=1, help="Detectar rostos a cada N frames.")
    parser.add_argument("--detect-scale", type=float, default=0.6, help="Escala usada para acelerar a deteccao.")
    parser.add_argument("--score-threshold", type=float, default=0.88, help="Confianca minima do detector facial YuNet.")
    parser.add_argument("--min-reliability", type=float, default=0.33, help="Confiabilidade minima da amostra facial.")
    parser.add_argument("--max-faces", type=int, default=24, help="Numero maximo de rostos marcados por frame.")
    parser.add_argument("--min-face-frac", type=float, default=0.08, help="Tamanho minimo do rosto como fracao da altura do quadro.")
    parser.add_argument("--min-skin-frac", type=float, default=0.33, help="Fracao minima de pele para validar um rosto.")
    parser.add_argument("--max-track-distance", type=float, default=100.0, help="Distancia maxima para manter o mesmo ID.")
    parser.add_argument("--max-missed", type=int, default=15, help="Frames sem deteccao antes de remover um rastreamento.")
    parser.add_argument("--min-draw-hits", type=int, default=2, help="Deteccoes confirmadas antes de desenhar a caixa.")
    parser.add_argument("--max-draw-missed", type=int, default=3, help="Frames sem deteccao antes de ocultar a caixa.")
    parser.add_argument("--max-frames", type=int, default=0, help="Processa apenas N frames; 0 processa o video inteiro.")
    parser.add_argument("--min-summary-samples", type=int, default=3, help="Minimo de amostras para entrar no CSV final.")
    return parser.parse_args()


if __name__ == "__main__":
    analyze_video(parse_args())
