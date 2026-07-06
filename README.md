# License Plate Detection — YOLOv5 + ONNX + Streamlit

A YOLOv5 object detector, fine-tuned to locate vehicle license plates, exported to ONNX, and served through a public Streamlit app.

**🔗 Live demo:** https://licenseplatedetectionyolov5-kxardcdpappnkapda7nzmfr.streamlit.app/

---

## What's in this repo

| Path | What it is |
|---|---|
| `License_Plate_Detection_YOLOv5__KD_.ipynb` | End-to-end notebook: dataset prep → training → inference → ONNX export |
| `streamlit_app/app.py` | Streamlit frontend — uploads an image, runs the ONNX model, draws detections |
| `streamlit_app/requirements.txt` | Python dependencies for the Streamlit app |
| `streamlit_app/best.onnx` | Trained model weights, exported from YOLOv5 to ONNX |

## Dataset

[Number Plate Detection](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection) on Kaggle, downloaded at runtime via `kagglehub`. Annotations are in PASCAL VOC XML format (one `.xml` per image).

> **Note:** the source annotations use two slightly different label strings for the same real-world object — `num_plate` and `number_plate` — so the trained model technically has `nc: 2`. Both labels mean "license plate." If retraining, consider normalizing these into a single class before building the YOLO config.

## Pipeline (see the notebook for full detail)

1. **Data preparation** — download the dataset, explore the PASCAL VOC annotations, visualise bounding boxes, split into train/val/test, and convert XML labels to YOLO's normalized `.txt` format.
2. **Training** — fine-tune `yolov5s.pt` (transfer learning from COCO) on the plate data.
3. **Inference & evaluation** — run the trained weights on held-out test images and compute precision/recall/mAP.
4. **Deployment** — export the trained weights to ONNX (`yolov5/export.py --include onnx`) and wrap them in a small Streamlit app for browser-based use, no PyTorch required at serving time.

## Running the notebook

Open `License_Plate_Detection_YOLOv5__KD_.ipynb` in Google Colab (GPU runtime recommended) and run top to bottom. It downloads the dataset, trains YOLOv5, evaluates it, and produces `streamlit_app/` (`app.py`, `requirements.txt`, `best.onnx`) ready to push to GitHub / deploy.

## Running the Streamlit app locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`, upload a photo of a vehicle, and adjust the confidence slider as needed.


## Tech stack

- [YOLOv5](https://github.com/ultralytics/yolov5) (Ultralytics) for training
- [ONNX](https://onnx.ai/) / [onnxruntime](https://onnxruntime.ai/) for framework-independent inference
- [Streamlit](https://streamlit.io/) for the web frontend
