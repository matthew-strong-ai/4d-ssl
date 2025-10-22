#!/usr/bin/env python3

"""
Standalone script to detect road signs in an image using:
- YOLOv8 (Ultralytics) – default COCO detector
- Grounding DINO v1 / v1.5 – open-vocabulary detection with text prompts

Usage:
  python detect_road_signs.py --image /path/to/image.jpg \
      [--detector yolov8|gdino1|gdino1_5] \
      [--model yolov8n.pt] [--conf 0.25] [--road_signs_only] \
      [--text_prompt "traffic sign . stop sign . speed limit sign . traffic light"] \
      [--gdino_config path/to/config.py] [--gdino_weights path/to/weights.pth] \
      [--box_threshold 0.3] [--text_threshold 0.25] \
      [--save out.jpg]

Notes:
- YOLOv8: COCO includes 'traffic light' and 'stop sign'. For broader sign categories, prefer Grounding DINO with a road-sign prompt.
- Grounding DINO: requires config and weights for the chosen version. See https://github.com/IDEA-Research/GroundingDINO
- Interactive viewer: press 'q' to quit, 's' to save (if --save not provided).
"""

import argparse
import os
import sys

import numpy as np

# Optional OpenCV import with helpful error
try:
    import cv2
except Exception:
    print("OpenCV (cv2) is required. Install with: pip install opencv-python")
    raise

# Detector choices
DETECTOR_CHOICES = ["yolov8", "gdino1", "gdino1_5"]

# Ultralytics YOLOv8 (optional)
_YOLO_AVAILABLE = True
try:
    from ultralytics import YOLO
except Exception:
    _YOLO_AVAILABLE = False

# Grounding DINO utils (optional)
_GDINO_AVAILABLE = True
try:
    from groundingdino.util.inference import load_model as gdino_load_model
    from groundingdino.util.inference import load_image as gdino_load_image
    from groundingdino.util.inference import predict as gdino_predict
    from groundingdino.util.inference import annotate as gdino_annotate
except Exception:
    _GDINO_AVAILABLE = False


COCO_CLASS_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# Indices for basic road-related sign classes in COCO
COCO_ROAD_SIGN_CLASS_IDS = {
    'traffic light': 9,
    'stop sign': 11,
}

# Default open-vocabulary prompt for road signs
DEFAULT_ROADSIGN_PROMPT = (
    "traffic light . road sign"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect road signs in an image with YOLOv8 or Grounding DINO")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")

    # Detector selection
    parser.add_argument("--detector", type=str, default="gdino1", choices=DETECTOR_CHOICES,
                        help="Detector backend to use")

    # YOLOv8 options
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model weights path or name")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., 'cpu', 'cuda', '0') for YOLO")
    parser.add_argument("--road_signs_only", action="store_true", help="Filter YOLO detections to road-sign classes")

    # Grounding DINO options
    parser.add_argument("--text_prompt", type=str, default=DEFAULT_ROADSIGN_PROMPT,
                        help="Open-vocabulary text prompt (phrases delimited by '.')")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="Grounding DINO box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Grounding DINO text threshold")
    parser.add_argument("--gdino_config", type=str, default=None, help="Path to Grounding DINO config.py")
    parser.add_argument("--gdino_weights", type=str, default=None, help="Path to Grounding DINO weights .pth")

    # Common visualization options
    parser.add_argument("--save", type=str, default=None, help="Optional path to save visualization")
    parser.add_argument("--auto_save", action="store_true", help="Automatically save output image with detections")
    parser.add_argument("--no_display", action="store_true", help="Skip interactive display, just save output")
    parser.add_argument("--line_thickness", type=int, default=2, help="Bounding box line thickness")
    parser.add_argument("--font_scale", type=float, default=0.6, help="Label font scale")
    return parser.parse_args()


def load_image_bgr(image_path: str) -> np.ndarray:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def filter_boxes_to_road_signs(result, class_name_list):
    # result: ultralytics Result object; result.boxes.cls is class indices
    if result.boxes is None or result.boxes.cls is None:
        return result
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    keep_ids = set(COCO_ROAD_SIGN_CLASS_IDS.values())
    mask = np.isin(cls, list(keep_ids))
    # Apply mask to boxes
    result.boxes = result.boxes[mask]
    return result


def draw_detections_yolo(image_bgr: np.ndarray, result, class_names, line_thickness=2, font_scale=0.6) -> np.ndarray:
    vis = image_bgr.copy()
    if result.boxes is None:
        return vis

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()  # [N, 4]
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones((len(boxes_xyxy),), dtype=float)
    classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros((len(boxes_xyxy),), dtype=int)

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, classes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = (0, 200, 0)  # green
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=line_thickness)
        label = f"{class_names[cls_id] if 0 <= cls_id < len(class_names) else cls_id}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(vis, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def run_yolov8(args, img_bgr: np.ndarray):
    if not _YOLO_AVAILABLE:
        raise ImportError("Ultralytics is not installed. Install with: pip install ultralytics")
    model = YOLO(args.model)
    results = model.predict(source=img_bgr, conf=args.conf, device=args.device, verbose=False)
    if len(results) == 0:
        print("No results returned by YOLO.")
        return img_bgr
    result = results[0]
    if args.road_signs_only:
        result = filter_boxes_to_road_signs(result, COCO_CLASS_NAMES)
    vis = draw_detections_yolo(img_bgr, result, COCO_CLASS_NAMES, line_thickness=args.line_thickness, font_scale=args.font_scale)
    return vis


def run_grounding_dino(args, image_path: str):
    if not _GDINO_AVAILABLE:
        raise ImportError(
            "groundingdino is not installed. Install with: pip install groundingdino and provide --gdino_config and --gdino_weights"
        )
    if not args.gdino_config or not args.gdino_weights:
        raise ValueError("Grounding DINO requires --gdino_config and --gdino_weights paths.")

    model = gdino_load_model(args.gdino_config, args.gdino_weights)
    image_source, image = gdino_load_image(image_path)
    boxes, logits, phrases = gdino_predict(
        model,
        image,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    # Annotate returns BGR image suitable for display
    vis = gdino_annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=phrases
    )
    return vis


def main():
    args = parse_args()


    # Read image (for YOLO); Grounding DINO uses its own loader but needs the path
    img_bgr = load_image_bgr(args.image)

    if args.detector == "yolov8":
        vis = run_yolov8(args, img_bgr)
    else:
        # Grounding DINO v1 or v1.5 share the same interface here
        # Users must provide matching config/weights for the chosen version
        vis = run_grounding_dino(args, args.image)

    # Determine save path
    save_path = args.save
    if args.auto_save and save_path is None:
        base, ext = os.path.splitext(args.image)
        save_path = f"{base}_{args.detector}_detections.png"
    
    # Save image if requested
    if save_path:
        ok = cv2.imwrite(save_path, vis)
        print(f"Saved visualization to: {save_path}" if ok else f"Failed to save visualization: {save_path}")
    
    # Interactive display (skip if --no_display is set)
    if not args.no_display:
        window_name = f"{args.detector.upper()} Road Sign Detection"
        cv2.imshow(window_name, vis)
        print("Press 'q' to quit, 's' to save visualization.")
        key = cv2.waitKey(0) & 0xFF

        # Save if user presses 's' and no save path was already set
        if key == ord('s') and not save_path:
            base, ext = os.path.splitext(args.image)
            interactive_save_path = f"{base}_{args.detector}_detections.png"
            ok = cv2.imwrite(interactive_save_path, vis)
            print(f"Saved visualization to: {interactive_save_path}" if ok else f"Failed to save visualization: {interactive_save_path}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 