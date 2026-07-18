# LiDAR Semantic Mapping Project

# 3D Semantic Mapping using LiDAR, RGB-D Cameras and Vision Foundation Models

A modular pipeline for building semantically enriched 3D maps by combining LiDAR/RGB-D data with modern computer vision and vision foundation models.

---

## Overview

Traditional point clouds only represent the geometry of an environment. This project aims to enrich those 3D representations with semantic information by identifying and segmenting objects such as chairs, tables, doors, and people.

The long-term goal is to generate an interactive semantic map that can support robotics, autonomous navigation, digital twins, and natural language scene understanding.

---

##  Objectives

- Acquire RGB-D and LiDAR data
- Generate dense point clouds
- Train object detection models
- Explore open-vocabulary object detection
- Perform instance segmentation
- Associate semantic information with 3D geometry
- Build a semantic map of the environment

---

## 🏗 Project Evolution

This project has evolved through multiple phases.

### Phase 1 – Sensor Setup

- Studied LiDAR fundamentals
- Understood Time-of-Flight (ToF) principle
- Explored Ouster SDK
- Configured development environment
- Connected to LiDAR sensor

### Phase 2 – Data Acquisition

- Captured RGB images
- Captured RGB-D data
- Recorded LiDAR scans
- Generated point cloud representations

### Phase 3 – Deep Learning Experiments

- Trained YOLOv8n
- Trained YOLOv8s
- Evaluated object detection performance
- Compared detection models

### Phase 4 – Foundation Models (Current)

- Integrated GroundingDINO
- Integrated Segment Anything (SAM)
- Designed a modular software architecture
- Implemented open-vocabulary object detection
- Implemented instance segmentation

### Phase 5 – Upcoming

- Semantic object representation
- Batch dataset processing
- Point cloud semantic fusion
- 3D semantic mapping
- Natural language scene querying

---

## Current Pipeline

RGB Image / RGB-D Frame / LiDAR Scan

↓

Data Acquisition

↓

Object Detection (GroundingDINO)

↓

Instance Segmentation (SAM)

↓

Semantic Objects

↓

3D Semantic Mapping (Upcoming)

---

##  Models Used

| Model | Purpose |
|--------|----------|
| Ouster SDK | LiDAR data acquisition |
| RGB-D Camera | RGB + depth capture |
| YOLOv8n | Initial object detection experiments |
| YOLOv8s | Improved object detection |
| GroundingDINO | Open-vocabulary object detection |
| Segment Anything (SAM) | Pixel-level segmentation |


## ✅ Features Implemented

- Modular project architecture
- GroundingDINO detector
- Segment Anything integration
- Open-vocabulary detection
- Instance segmentation
- Visualization module
- Detection pipeline

---

## 🚧 Future Work

- Custom visualization engine
- Semantic object abstraction
- Dataset-wide inference
- Point cloud semantic fusion
- Interactive semantic maps
- Natural language querying

---

##  Technologies

- Python
- PyTorch
- OpenCV
- Ouster SDK
- GroundingDINO
- Segment Anything
- YOLOv8
- NumPy

---

## 📸 Results

*(Screenshots of LiDAR scans, YOLO detections, GroundingDINO detections, SAM masks, and future semantic maps will be added here.)*

---

## 👤 Author

**Harman Kaur**

B.Tech Computer and Communication Engineering  
Manipal University Jaipur
