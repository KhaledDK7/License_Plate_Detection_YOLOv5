
import io
import time
import os
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")
CLASS_NAMES = ["licence"]        
DEFAULT_CONF = 0.40
IOU_THRESHOLD = 0.45

st.set_page_config(page_title="Plate Finder", page_icon="🚗", layout="centered")



@st.cache_resource
def load_model(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_size = input_meta.shape[-1] if isinstance(input_meta.shape[-1], int) else 416
    return session, input_size



def letterbox(img: Image.Image, size: int):
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2

    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114)) 
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def image_to_tensor(img: Image.Image):
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)



def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, iou_thresh):
    boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)
    keep = []
    for b in boxes:
        if all(not (b["cls"] == k["cls"] and iou(b["xyxy"], k["xyxy"]) > iou_thresh) for k in keep):
            keep.append(b)
    return keep


def decode_output(output, conf_thresh, scale, pad_x, pad_y):

    preds = output[0]
    expected_dim = 5 + len(CLASS_NAMES)
    if preds.shape[-1] != expected_dim and preds.shape[0] == expected_dim:
        preds = preds.T
    num_classes = preds.shape[1] - 5

    objectness = preds[:, 4]
    class_scores = preds[:, 5:5 + num_classes]
    best_cls = np.argmax(class_scores, axis=1)
    best_score = class_scores[np.arange(len(preds)), best_cls]
    conf = objectness * best_score

    mask = conf > conf_thresh
    boxes = []
    for row, cls_id, score in zip(preds[mask], best_cls[mask], conf[mask]):
        cx, cy, w, h = row[:4]
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale
        boxes.append({"xyxy": (x1, y1, x2, y2), "score": float(score), "cls": int(cls_id)})
    return boxes


def draw_boxes(img: Image.Image, boxes):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    line_w = max(2, img.width // 250)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, img.width // 45))
    except OSError:
        font = ImageFont.load_default()

    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        label = f'{CLASS_NAMES[b["cls"]]} {b["score"] * 100:.0f}%'
        draw.rectangle([x1, y1, x2, y2], outline="#f4b400", width=line_w)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0] + 10, text_bbox[3] - text_bbox[1] + 8
        draw.rectangle([x1 - line_w / 2, y1 - text_h, x1 - line_w / 2 + text_w, y1], fill="#f4b400")
        draw.text((x1 + 5, y1 - text_h + 2), label, fill="#16241f", font=font)
    return img

# UI
st.title("🚗 Plate Finder")
st.caption("YOLOv5, exported to ONNX, running server-side via onnxruntime.")

try:
    session, model_input_size = load_model(MODEL_PATH)
    st.success(f"Model loaded — input size {model_input_size}×{model_input_size}", icon="✅")
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`. Make sure it sits next to this script.\n\n{e}")
    st.stop()

conf_thresh = st.slider("Confidence threshold", 0.05, 0.90, DEFAULT_CONF, 0.05)

uploaded_file = st.file_uploader("Upload a photo of a vehicle", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    t0 = time.time()
    canvas, scale, pad_x, pad_y = letterbox(img, model_input_size)
    tensor = image_to_tensor(canvas)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: tensor})[0]

    boxes = decode_output(output, conf_thresh, scale, pad_x, pad_y)
    boxes = nms(boxes, IOU_THRESHOLD)
    elapsed_ms = (time.time() - t0) * 1000

    annotated = draw_boxes(img, boxes)
    st.image(annotated, use_container_width=True)
    st.caption(f"{len(boxes)} detection(s) · {elapsed_ms:.0f} ms")

    if boxes:
        st.subheader("Detections")
        st.table([
            {
                "class": CLASS_NAMES[b["cls"]],
                "confidence": f'{b["score"] * 100:.1f}%',
                "box (x1, y1, x2, y2)": tuple(round(v, 1) for v in b["xyxy"]),
            }
            for b in boxes
        ])
    else:
        st.info("No plate found above the current confidence threshold — try lowering it.")
else:
    st.info("Upload an image to run detection.")
