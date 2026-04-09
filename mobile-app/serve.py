"""HTTPS server with server-side YOLOv8 inference endpoint.

Usage:
    python serve.py

Then open https://<your-ip>:8443 on your phone (accept the self-signed cert warning).
"""

import http.server
import json
import os
import ssl
import subprocess
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

PORT_HTTPS = 8443
CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"
MODEL_PATH = "model/best.onnx"
CLASSES_PATH = "model/classes.json"
MODEL_INPUT_SIZE = 320
IOU_THRESHOLD = 0.5

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

# ─── Load model once at startup ───────────────────────────────
print(f"Loading ONNX model from {MODEL_PATH}...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"Model loaded. Input: {input_name} {session.get_inputs()[0].shape}")


def preprocess(jpeg_bytes):
    """Decode JPEG and convert to NCHW float32 tensor."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img.shape[0] != MODEL_INPUT_SIZE or img.shape[1] != MODEL_INPUT_SIZE:
        img = cv2.resize(img, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img, axis=0)   # add batch dim


def postprocess(output, conf_threshold):
    """Extract detections from YOLOv8 output [1, 5, 2100]."""
    data = output[0]  # [5, 2100]
    num_classes = data.shape[0] - 4
    detections = []
    for i in range(data.shape[1]):
        scores = data[4:, i]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        if score < conf_threshold:
            continue
        cx, cy, w, h = data[0, i], data[1, i], data[2, i], data[3, i]
        detections.append({
            "x1": float(cx - w / 2),
            "y1": float(cy - h / 2),
            "x2": float(cx + w / 2),
            "y2": float(cy + h / 2),
            "score": score,
            "classId": class_id,
        })
    # NMS
    if not detections:
        return detections
    detections.sort(key=lambda d: d["score"], reverse=True)
    keep = []
    removed = set()
    for i, a in enumerate(detections):
        if i in removed:
            continue
        keep.append(a)
        for j in range(i + 1, len(detections)):
            if j in removed:
                continue
            b = detections[j]
            if a["classId"] == b["classId"] and iou(a, b) > IOU_THRESHOLD:
                removed.add(j)
    return keep


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)


class InferenceHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path.startswith("/infer"):
            self.handle_infer()
        else:
            self.send_error(404)

    def handle_infer(self):
        # Parse confidence threshold from query string
        conf = 0.45
        if "?" in self.path:
            for part in self.path.split("?")[1].split("&"):
                if part.startswith("conf="):
                    conf = float(part.split("=")[1])

        content_length = int(self.headers["Content-Length"])
        jpeg_bytes = self.rfile.read(content_length)

        t0 = time.time()
        tensor = preprocess(jpeg_bytes)
        results = session.run(None, {input_name: tensor})
        detections = postprocess(results[0], conf)
        dt = (time.time() - t0) * 1000

        response = json.dumps({
            "detections": detections,
            "inferenceMs": round(dt, 1),
        }).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        # Suppress GET logs, keep POST logs
        if "POST" in str(args):
            super().log_message(format, *args)


def generate_self_signed_cert():
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return
    print("Generating self-signed certificate for HTTPS...")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", KEY_FILE, "-out", CERT_FILE,
        "-days", "365", "-nodes",
        "-subj", "/CN=localhost",
    ], check=True)


def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    generate_self_signed_cert()
    ip = get_local_ip()

    print(f"\n  Network (HTTPS): https://{ip}:{PORT_HTTPS}")
    print(f"\n  On your phone, open the HTTPS URL and accept the certificate warning.\n")

    server = http.server.HTTPServer(("0.0.0.0", PORT_HTTPS), InferenceHandler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)
    server.socket = context.wrap_socket(server.socket, server_side=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
