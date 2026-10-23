"""CPU runtime for the measured YOLO26n orange-center model.

The exported model was inspected locally and returns end-to-end NMS rows with
shape ``[1, 300, 6]`` and columns ``x1,y1,x2,y2,confidence,class``.  This module
validates that contract at runtime instead of assuming a YOLOv8-style tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class YoloDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def center(self):
        return 0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2)


@dataclass
class YoloResult:
    frame_id: int
    captured_at: float
    finished_at: float
    detections: list[YoloDetection]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    raw_bgr: object
    error: str = ""

    @property
    def total_ms(self):
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


class OrangeYoloDetector:
    """One-time model loader and exact letterbox-aware detector."""

    def __init__(
        self,
        model_path: str,
        confidence: float,
        input_size: int = 640,
        min_box_px: float = 3.0,
        max_box_area_frac: float = 0.65,
        threads: int = 2,
    ):
        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self.input_size = int(input_size)
        self.min_box_px = float(min_box_px)
        self.max_box_area_frac = float(max_box_area_frac)
        self.threads = max(1, int(threads))
        self.runtime_name = ""
        self.net = None
        self._ort_session = None
        self._ort_input_name = None
        errors = []

        # The PhysiCar/evaluator image currently provides ONNX Runtime 1.29.
        # Prefer it because OpenCV 4.11 rejects YOLO26's legal TopK(K=300)
        # graph, while OpenCV 5 accepts it.  No package is installed at launch.
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                self.model_path, sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or list(inputs[0].shape) != [1, 3, self.input_size, self.input_size]:
                raise RuntimeError(f"unexpected ONNX input metadata: {[x.shape for x in inputs]}")
            if len(outputs) != 1 or list(outputs[0].shape) != [1, 300, 6]:
                raise RuntimeError(f"unexpected ONNX output metadata: {[x.shape for x in outputs]}")
            self._ort_session = session
            self._ort_input_name = inputs[0].name
            self.runtime_name = f"onnxruntime-{ort.__version__}"
        except Exception as exc:
            errors.append(f"onnxruntime:{type(exc).__name__}:{exc}")

        if self._ort_session is None:
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.runtime_name = f"opencv-dnn-{cv2.__version__}"
            except Exception as exc:
                errors.append(f"opencv:{type(exc).__name__}:{exc}")
                raise RuntimeError("; ".join(errors)) from exc

    def _letterbox(self, raw_bgr):
        h, w = raw_bgr.shape[:2]
        size = self.input_size
        scale = min(size / float(w), size / float(h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(raw_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = size - nw, size - nh
        left = int(round(dw / 2.0 - 0.1))
        right = dw - left
        top = int(round(dh / 2.0 - 0.1))
        bottom = dh - top
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        blob = cv2.dnn.blobFromImage(
            padded, scalefactor=1.0 / 255.0,
            size=(size, size), swapRB=True, crop=False,
        )
        return blob, scale, float(left), float(top)

    @staticmethod
    def _rows(output):
        rows = np.asarray(output)
        if rows.ndim == 3 and rows.shape[0] == 1:
            rows = rows[0]
        if rows.ndim != 2 or rows.shape[1] != 6:
            raise RuntimeError(
                f"unsupported orange YOLO output shape {tuple(np.asarray(output).shape)}; "
                "expected measured end-to-end [1,N,6]"
            )
        return rows

    def infer(self, raw_bgr, frame_id: int = 0, captured_at: Optional[float] = None):
        captured_at = time.monotonic() if captured_at is None else float(captured_at)
        raw_copy = np.ascontiguousarray(raw_bgr)
        h, w = raw_copy.shape[:2]

        t0 = time.perf_counter()
        blob, scale, pad_x, pad_y = self._letterbox(raw_copy)
        t1 = time.perf_counter()
        if self._ort_session is not None:
            output = self._ort_session.run(None, {self._ort_input_name: blob})[0]
        else:
            self.net.setInput(blob)
            output = self.net.forward()
        t2 = time.perf_counter()

        detections = []
        for row in self._rows(output):
            conf = float(row[4])
            cls = int(round(float(row[5])))
            if conf < self.confidence or cls != 0:
                continue
            x1 = (float(row[0]) - pad_x) / scale
            y1 = (float(row[1]) - pad_y) / scale
            x2 = (float(row[2]) - pad_x) / scale
            y2 = (float(row[3]) - pad_y) / scale
            x1, x2 = sorted((max(0.0, min(float(w - 1), x1)), max(0.0, min(float(w - 1), x2))))
            y1, y2 = sorted((max(0.0, min(float(h - 1), y1)), max(0.0, min(float(h - 1), y2))))
            bw, bh = x2 - x1, y2 - y1
            if bw < self.min_box_px or bh < self.min_box_px:
                continue
            if bw * bh > self.max_box_area_frac * float(w * h):
                continue
            detections.append(YoloDetection(x1, y1, x2, y2, conf, cls))
        t3 = time.perf_counter()
        return YoloResult(
            frame_id=int(frame_id), captured_at=captured_at,
            finished_at=time.monotonic(), detections=detections,
            preprocess_ms=(t1 - t0) * 1000.0,
            inference_ms=(t2 - t1) * 1000.0,
            postprocess_ms=(t3 - t2) * 1000.0,
            raw_bgr=raw_copy,
        )


class AsyncOrangeYolo:
    """Latest-frame worker that never blocks the stable 16 Hz control loop."""

    def __init__(self, detector: OrangeYoloDetector, every_n: int):
        self.detector = detector
        self.every_n = max(1, int(every_n))
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker, name="orange-yolo", daemon=True
        )
        self._thread.start()

    def submit(self, frame_id: int, captured_at: float, raw_bgr):
        if self._closed or int(frame_id) % self.every_n:
            return False
        item = (int(frame_id), float(captured_at), raw_bgr.copy())
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                return False
        return True

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame_id, captured_at, raw = item
            try:
                result = self.detector.infer(raw, frame_id, captured_at)
            except Exception as exc:
                result = YoloResult(
                    frame_id, captured_at, time.monotonic(), [],
                    0.0, 0.0, 0.0, raw, error=f"{type(exc).__name__}: {exc}",
                )
            with self._lock:
                self._latest = result

    def latest(self):
        with self._lock:
            return self._latest

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=2.0)
