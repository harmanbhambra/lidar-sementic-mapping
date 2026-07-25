# Open-Vocabulary Semantic Scene Understanding for Indoor Environments

An indoor semantic perception system exploring the transition from **closed-vocabulary object detection** to **open-vocabulary scene understanding**.

The project began with RGB-D data acquisition using an Intel RealSense camera and closed-vocabulary experiments using **YOLOv8n and YOLOv8s**, and evolved into an open-vocabulary perception pipeline combining **GroundingDINO** and **Segment Anything (SAM)**.

The current system accepts an RGB image and a text-defined object vocabulary and produces object detections, segmentation masks, annotated images, and structured semantic JSON without task-specific retraining for newly requested concepts.

**Core Stack:** Python · PyTorch · GroundingDINO · SAM · YOLOv8 · OpenCV · ROS2 · Intel RealSense · RTAB-Map

---

## Overview

The project explores how an indoor perception system can move beyond a predefined set of object classes.

The development progressed through three major stages:

1. **RGB-D Data Acquisition** — RealSense, ROS2 and RTAB-Map experimentation
2. **Closed-Vocabulary Detection** — training and evaluating YOLOv8n and YOLOv8s
3. **Open-Vocabulary Semantic Perception** — GroundingDINO + SAM with structured scene representations

The final perception pipeline is:

```text
                    Text Prompt
                        |
                        v
RGB Image ------> GroundingDINO
                        |
                        v
             Boxes + Labels + Scores
                        |
                        v
                       SAM
                        |
                        v
               Segmentation Masks
                        |
                        v
                  DetectedObject
                        |
                        v
                      Scene
                   /         \
                  v           v
         Annotated Image   Semantic JSON
                               |
                               v
                           Evaluation
```

---

## 1. RGB-D Data Acquisition

The original project direction investigated semantic mapping using RGB-D sensing.

Sensor experimentation was carried out using an **Intel RealSense D435i**, **ROS2 Humble**, and **RTAB-Map**.

The RGB-D acquisition components support:

- RGB frames
- aligned depth frames
- camera intrinsics
- synchronized sensor observations
- RGB-D dataset recording

The dataset recording utility can store:

```text
RGB frames
Depth frames
Timestamps
Camera intrinsics
```

This work provides the sensor foundation required for future projection of semantic information from 2D image coordinates into 3D space.

> The RGB-D sensor acquisition and RealSense/ROS2 setup were developed collaboratively during the sensor experimentation phase of the project.

---

## 2. Closed-Vocabulary Baseline — YOLOv8

The indoor dataset was initially used to train and evaluate conventional object detectors using:

- **YOLOv8n**
- **YOLOv8s**

This established a closed-vocabulary baseline:

```text
Annotated Indoor Dataset
          |
          v
   YOLOv8n / YOLOv8s
          |
          v
       Training
          |
          v
Detection of predefined classes
```

These experiments provided experience with dataset preparation, model training, inference, and evaluation.

However, conventional supervised object detection is constrained by its predefined class vocabulary. Supporting additional object concepts generally requires appropriate labeled data and further training.

This motivated the transition toward **open-vocabulary detection**.

---

## 3. Open-Vocabulary Semantic Pipeline

### GroundingDINO

The main pipeline uses **GroundingDINO** for text-conditioned object detection.

Instead of relying only on a predefined task-specific class list, the desired object concepts are provided through a text prompt at inference time.

For example:

```python
text_prompt = (
    "person . chair . table . door . window . "
    "backpack . bottle . laptop . bench . "
    "fire extinguisher . board"
)
```

GroundingDINO returns:

- object labels
- confidence scores
- bounding boxes

The vocabulary can be changed for another scene without retraining the detector for the newly requested concepts.

### Segment Anything (SAM)

Each GroundingDINO bounding box is passed to **SAM ViT-H** to obtain a pixel-level segmentation mask.

```text
GroundingDINO Detection
          |
          v
     Bounding Box
          |
          v
         SAM
          |
          v
 Segmentation Mask
```

This provides a more detailed representation of object regions than bounding boxes alone.

---

## 4. Semantic Scene Representation

Detection and segmentation results are combined into custom semantic structures.

Each detection becomes a `DetectedObject` containing:

```text
DetectedObject
├── label
├── confidence
├── bounding_box
└── segmentation mask
```

For example:

```python
DetectedObject(
    label="person",
    confidence=0.72,
    bounding_box=...,
    mask=...
)
```

All detected objects from an image are then grouped into a `Scene`:

```python
Scene(
    image=image_source,
    objects=detected_objects
)
```

The `Scene` representation supports operations such as:

- counting objects
- retrieving detected labels
- filtering objects by label
- visualization
- JSON serialization

Each processed image can therefore produce both:

```text
Annotated Image
      +
Semantic JSON
```

The JSON representation allows the semantic output to be used programmatically rather than existing only as a visualization.

---

## 5. Batch Processing and Evaluation

The pipeline was extended from single-image inference to multi-image batch processing.

For every image:

```text
Input Image
     |
     v
GroundingDINO
     |
     v
SAM
     |
     v
Semantic Scene
    /     \
   v       v
Image     JSON
```

A custom `SceneEvaluator` was also implemented to aggregate experiment-level statistics including:

- images processed
- total detections
- average objects per image
- average confidence
- label frequencies

This makes it possible to analyze larger experiments without manually inspecting every semantic JSON file.

---

## 6. 25-Image Indoor Experiment

The open-vocabulary pipeline was evaluated on 25 indoor images using an expanded vocabulary containing concepts such as:

```text
person
chair
table
door
window
backpack
bottle
laptop
bench
fire extinguisher
board
```

Example detection counts observed during the experiment included:

| Detected Phrase | Count |
|---|---:|
| Door | 18 |
| Window | 13 |
| Laptop | 9 |
| Bench | 8 |
| Chair | 6 |
| Backpack | 6 |
| Fire extinguisher | 3 |
| Bottle | 3 |
| Board | 3 |
| Table bench | 2 |
| Chair bench | 1 |

These values represent **model detections rather than ground-truth accuracy measurements**.

### Observations

The experiment demonstrated successful detection and segmentation of several indoor object categories.

It also revealed limitations including:

- duplicate detections
- occasional false positives
- missed objects
- lower confidence for difficult or distant objects
- semantic ambiguity between related concepts

For example, visually related seating concepts sometimes produced labels such as:

```text
chair
bench
bench seating
chair bench
```

This demonstrates both the flexibility and ambiguity that can occur with text-conditioned detection.

---

## 7. Generalization Experiment

To test whether the pipeline was dependent on the original indoor dataset, it was evaluated on a visually different indoor scene.

A new text vocabulary was provided:

```text
vase . lamp . bed . chair . mirror . window . flower pot . table
```

No task-specific retraining was performed.

The pipeline produced **14 detections**:

| Object | Confidence |
|---|---:|
| Mirror | 0.79 |
| Lamp | 0.78 |
| Lamp | 0.73 |
| Window | 0.67 |
| Window | 0.59 |
| Vase | 0.57 |
| Chair | 0.54 |
| Table | 0.54 |
| Table | 0.50 |
| Window | 0.50 |
| Bed | 0.44 |
| Flower pot | 0.41 |
| Vase | 0.37 |
| Bed | 0.36 |

This demonstrates that the same GroundingDINO + SAM pipeline can process an unseen environment with a newly specified object vocabulary without task-specific retraining.

Qualitative inspection also showed duplicate detections and occasional false positives, highlighting areas for future improvement.

---

## 8. Project Evolution

```text
             RGB-D Sensor Acquisition
          RealSense + ROS2 + RTAB-Map
                       |
                       v
            Indoor Dataset Experiments
                       |
                       v
          Closed-Vocabulary Detection
               YOLOv8n / YOLOv8s
                       |
                       v
          Open-Vocabulary Detection
                  GroundingDINO
                       |
                       v
               SAM Segmentation
                       |
                       v
            Semantic Representation
             DetectedObject + Scene
                       |
                 +-----+-----+
                 |           |
                 v           v
            Visualization   JSON
                 \           /
                  \         /
                   v       v
                   Evaluation
                       |
                       v
               Generalization Test
```

---

## 9. Repository Structure

```text
lidar-sementic-mapping/
│
├── src/
│   ├── detectors/
│   │   └── grounding_dino.py
│   │
│   ├── segmentation/
│   │   └── sam_segmenter.py
│   │
│   ├── semantic/
│   │   └── ...
│   │
│   ├── visualization/
│   │   └── visualizer.py
│   │
│   ├── evaluation/
│   │   └── ...
│   │
│   ├── sensors/
│   │   └── realsense_interface.py
│   │
│   └── main.py
│
├── scripts/
│   ├── evaluate_results.py
│   ├── test_generalization.py
│   ├── save_rgbd_dataset.py
│   ├── train_yolo.py
│   ├── train_yolov8s.py
│   ├── test_yolo.py
│   ├── test_yolov8s.py
│   └── ...
│
├── data/
│   └── generalization_test/
│
├── results/
├── models/
├── GroundingDINO/
├── requirements.txt
└── README.md
```

Large datasets, model checkpoints, generated results, and virtual environments may be excluded from version control.

---

## 10. Technologies

### Computer Vision and Machine Learning

- Python
- PyTorch
- GroundingDINO
- Segment Anything (SAM ViT-H)
- YOLOv8n
- YOLOv8s
- OpenCV
- NumPy

### Robotics and Sensor Experimentation

- Intel RealSense D435i
- RGB-D imaging
- ROS2 Humble
- RTAB-Map

### Development

- Git
- GitHub
- Python virtual environments

---

## 11. Running the Pipeline

Activate the project environment.

Example on Windows PowerShell:

```powershell
.\semantic-v2\Scripts\Activate.ps1
```

Run the main semantic pipeline:

```powershell
python src/main.py
```

Generated outputs are written to the configured experiment folder under:

```text
results/
```

and can include:

```text
detected_<image>.jpg
semantic_<image>.json
```

### Evaluate an Experiment

```powershell
python scripts/evaluate_results.py
```

### Run the Generalization Test

Place the test image in:

```text
data/generalization_test/
```

Configure the desired vocabulary inside:

```text
scripts/test_generalization.py
```

and run:

```powershell
python scripts/test_generalization.py
```

---

## 12. Limitations and Future Work

The current GroundingDINO + SAM semantic pipeline operates primarily on **2D RGB images**.

Although RGB-D acquisition and RTAB-Map experimentation were performed, the current semantic masks are not yet fused with depth, camera intrinsics, and camera poses into a persistent 3D semantic map.

The natural next stage is:

```text
RGB + Depth
     |
     v
Semantic Detection
     |
     v
Segmentation
     |
     v
Depth + Intrinsics
     |
     v
3D Projection
     |
     v
Semantic Point Cloud
     |
     v
Camera Pose + Multi-frame Fusion
     |
     v
Persistent 3D Semantic Map
     |
     v
Natural-Language Spatial Queries
```

Future improvements include:

- semantic mask projection using depth
- point-cloud generation
- camera-pose integration
- multi-frame semantic fusion
- object association and tracking
- duplicate detection suppression
- quantitative evaluation using ground-truth annotations
- GPU acceleration
- natural-language semantic map querying

Potential future queries could include:

```text
"Where is the chair?"
```

or:

```text
"Locate the chair next to the table."
```

---

## Collaboration and Acknowledgements

The RGB-D sensor acquisition, Intel RealSense setup, and associated ROS2 experimentation were developed collaboratively during the sensor-development phase of the project.

This project builds upon technologies and open-source work including:

- GroundingDINO
- Segment Anything (SAM)
- Ultralytics YOLO
- Intel RealSense
- ROS2
- RTAB-Map

---

## Project Status

### V1 — Open-Vocabulary Semantic Perception

**Completed**

- [x] RGB-D sensor acquisition experiments
- [x] RealSense RGB-D interface
- [x] RGB-D dataset recording
- [x] YOLOv8n closed-vocabulary training
- [x] YOLOv8s closed-vocabulary training
- [x] GroundingDINO open-vocabulary detection
- [x] Dynamic text-prompt vocabulary
- [x] SAM segmentation
- [x] Semantic `DetectedObject` representation
- [x] Semantic `Scene` representation
- [x] Annotated visualization
- [x] Semantic JSON export
- [x] Multi-image batch processing
- [x] Experiment-level evaluation
- [x] 25-image indoor experiment
- [x] Out-of-dataset generalization test

**Future**

- [ ] 3D semantic projection
- [ ] Camera-pose integration
- [ ] Multi-frame semantic fusion
- [ ] Persistent 3D semantic mapping
- [ ] Natural-language spatial querying
