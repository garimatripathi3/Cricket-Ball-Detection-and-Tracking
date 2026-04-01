# Cricket Ball Detection and Tracking

This project implements a complete end-to-end pipeline for detecting and tracking a cricket ball in fixed camera videos using YOLOv8 and motion-based tracking (Kalman Filter).

The system outputs:

* Ball centroid per frame
* Visibility flag (1 / 0)
* Processed video with trajectory overlay

## Problem Overview

Tracking a cricket ball is challenging due to:

* Extremely small object size
* High-speed motion
* Motion blur
* Background confusion (pitch, players, lighting variations)

This project combines detection and motion modeling to achieve stable and robust tracking.

## Approach

### Detection

* YOLOv8-based detector
* High-resolution input (imgsz = 960) for small object detection
* Confidence and class-based filtering

### Tracking

* Kalman Filter for motion prediction
* State representation: position and velocity (x, y, vx, vy)
* Handles missed detections and smooths trajectory

### Filtering and Tracking Logic

* Motion gating to reject unrealistic jumps
* Direction consistency scoring
* Size-based filtering
* Adaptive confidence threshold

### ROI Optimization

* Fixed region of interest focused on pitch area
* Reduces false positives and improves efficiency

## Project Structure

```
project/
├── code/
│   ├── config.py
│   ├── detector.py
│   ├── kalman.py
│   ├── tracker.py
│   ├── pipeline.py
│   ├── visualization.py
│   ├── utils.py
│   ├── main_inference.py
│   ├── train.ipynb
│   └── evaluate.ipynb
├── videos/
├── weight/
│   └── best.pt
├── test_results/
│   ├── annotations/
│   ├── debug/
│   ├── videos/
├── evaluation_results/
│   ├── annotated_examples/
│   ├── annotated_grid.png
│   ├── evaluation_summary.csv
│   ├── per_image_results.csv
├── README.md
├── requirements.txt
└── report.pdf
```

## Training

Training is performed using YOLOv8 with a configuration optimized for small object detection.

```python
results = model.train(
    data="data.yaml",
    epochs=120,
    imgsz=960,
    batch=8,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.1,
    warmup_epochs=3,
    weight_decay=0.0005,
    patience=20,
    mosaic=0.5,
    mixup=0.0,
    scale=0.5,
    translate=0.1,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=2.0
)
```

Run training using:

```bash
jupyter notebook code/train.ipynb
```

## Inference and Tracking

Run the full pipeline:

```bash
python code/main_inference.py
```

Ensure the trained model is placed at:

```
weight/best.pt
```

## Evaluation

Evaluation is performed on a labeled dataset in an image-wise manner to assess both detection and localization performance.

### Metrics

* **Visibility Accuracy**
  Measures how often the model correctly predicts whether the ball is visible.

* **Precision (Visible Class)**
  Fraction of predicted visible frames that are correct.

* **Recall (Visible Class)**
  Fraction of actual visible frames that are correctly detected.

* **Centroid Error (Euclidean Distance)**
  Distance between predicted and ground truth ball center:

  [
  d = \sqrt{(x_{pred} - x_{gt})^2 + (y_{pred} - y_{gt})^2}
  ]

  Reported as:

  * Mean centroid error
  * Median centroid error

### Outputs

The evaluation script generates:

* `evaluation_results/per_image_results.csv`
* `evaluation_results/evaluation_summary.csv`
* `evaluation_results/annotated_examples/`

### Notes

Ground truth annotations are not available for the test videos. Therefore, quantitative evaluation is performed on a labeled dataset, while video level performance is demonstrated qualitatively through trajectory outputs.

Example annotated frames in `evaluation_results/annotated_examples/` are sampled from the evaluation dataset for qualitative analysis.

## Output Format

### CSV Format

```
frame,x,y,visible
```

Example:

```
0,512,320,1
1,520,330,1
2,-1,-1,0
```

* `visible = 1` → ball detected
* `visible = 0` → ball not visible

## Output Details

* Detected ball centroid is rendered as a point
* Predicted position is shown during missed detections
* Trajectory trail visualizes motion over time
* ROI boundary is displayed for reference

## Key Design Decisions

**High Resolution (960)**
Improves detection of extremely small objects.

**AdamW Optimizer**
Provides stable convergence and better generalization.

**No Mixup Augmentation**
Avoids distortion of small object features.

**Kalman Filter (instead of deep tracking)**
Appropriate for single-object tracking; efficient and robust to missed detections.

**Centroid Tracking**
More stable than bounding box tracking for small objects.

**No Training on Live Match Videos**
Prevents overfitting to broadcast-specific patterns and ensures the model learns cricket ball dynamics rather than match-specific visual artifacts.

## Limitations

* Missed detections under extreme motion blur
* Occasional false positives in complex scenes
* Limited dataset diversity


## Future Improvements

* Larger and more diverse dataset
* Motion blur augmentation
* Physics-based trajectory modeling
* Multi-frame detection fusion

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes

* The system prioritizes robust tracking over raw detection accuracy
* Designed to generalize across different cricket scenarios# Cricket-Ball-Tracking-
