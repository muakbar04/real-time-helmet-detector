# Helmet Detection System with YOLOv11

An end-to-end, production-ready pipeline for detecting helmets using the YOLOv11 architecture. This repository contains a comprehensive Google Colab notebook that walks you through each step of the development process—from verifying the hardware environment and setting up dependencies, to acquiring datasets, training the model, and deploying it for real-time detection via webcam or video stream. The system leverages the robust capabilities of YOLOv11 for accurate, fast, and scalable object detection.

---

## Table of Contents

1. [Verify GPU Environment](#1-verify-gpu-environment)
2. [Install Ultralytics YOLO Package](#2-install-ultralytics-yolo-package)
3. [Sanity Check & Dependency Validation](#3-sanity-check--dependency-validation)
4. [Import Core APIs & Visualization Tools](#4-import-core-apis--visualization-tools)
5. [Acquire Dataset via Roboflow SDK](#5-acquire-dataset-via-roboflow-sdk)
6. [Train YOLOv11‑Nano Model](#6-train-YOLOv11nano-model)
7. [Analyze Confusion Matrix](#7-analyze-confusion-matrix)
8. [Evaluate Label Distribution](#8-evaluate-label-distribution)
9. [Track Loss & mAP Metrics](#9-track-loss--map-metrics)
10. [Inspect Augmented Training Samples](#10-inspect-augmented-training-samples)
11. [Perform Batch Inference on Test Set](#11-perform-batch-inference-on-test-set)
12. [Single-Image Inference Example](#12-single-image-inference-example)
13. [Video Stream Processing](#13-video-stream-processing)
14. [Real‑Time Webcam Deployment](#15-real-time-webcam-deployment)

---

## 1. Verify GPU Environment

**Objective:** Confirm the presence of a CUDA-enabled NVIDIA GPU and sufficient VRAM.
**Why:** Ensures high-throughput training and inference performance, prevents runtime bottlenecks.
**Outcome:** Hardware report with GPU model, driver version, and memory utilization.

---

## 2. Install Ultralytics YOLO Package

**Objective:** Install the official Ultralytics library for YOLOv11.
**Why:** Provides a unified CLI and Python API for model operations—training, evaluation, and export.
**Outcome:** Access to `yolo` commands and Python classes for seamless workflow integration.

---

## 3. Sanity Check & Dependency Validation

**Objective:** Run `ultralytics.checks()` to validate Python, PyTorch, and CUDA configurations.
**Why:** Verifies compatibility of critical deep learning frameworks and prevents misconfiguration errors.
**Outcome:** Detailed report on versions, GPU availability, and missing/incorrect dependencies.

---

## 4. Import Core APIs & Visualization Tools

**Objective:** Load the `YOLO` class and Jupyter display utilities.
**Why:** Establishes the core interfaces for model manipulation and inline result rendering.
**Outcome:** Ready-to-use Python objects for training workflows and image/video display.

---

## 5. Acquire Dataset via Roboflow SDK

**Objective:** Programmatically download the helmet detection dataset in YOLOv11 format.
**Why:** Automates dataset retrieval, ensures correct directory structure, and standardizes splits.
**Outcome:** Local directory containing `data.yaml`, and `train/`, `valid/`, `test/` partitions with images and labels.

---

## 6. Train YOLOv11‑Nano Model

**Objective:** Execute a high-performance training session using the YOLOv11‑Nano variant.
**Why:** Balances model size and speed for rapid iteration, suitable for limited-resource environments.
**Key Parameters:**

* `data` path to dataset config
* Pretrained `YOLOv11n.pt` weights
* `epochs` count for convergence
* `imgsz` resolution
  **Outcome:** Trained model checkpoints saved to `runs/detect` directory.

---

## 7. Analyze Confusion Matrix

**Objective:** Visualize class-level performance via a confusion matrix.
**Why:** Identifies high-error classes for targeted data augmentation or rebalancing.
**Outcome:** PNG plot showing true positives, false positives, and false negatives per class.

---

## 8. Evaluate Label Distribution

**Objective:** Plot the frequency of bounding-box labels across all classes.
**Why:** Detects dataset imbalance, informs oversampling or augmentation strategies.
**Outcome:** Bar chart illustrating the number of annotated instances per class.

---

## 9. Track Loss & mAP Metrics

**Objective:** Display training/validation loss curves and mean Average Precision (mAP).
**Why:** Monitors convergence behavior, detects overfitting, and measures detection quality.
**Outcome:** Line plots for classification, localization, and DFL losses, alongside mAP\@0.5 and mAP\@0.5:0.95.

---

## 10. Inspect Augmented Training Samples

**Objective:** Preview the first batch of augmented images and bounding boxes.
**Why:** Validates augmentation pipelines (mosaic, flips, color transforms) and label alignment.
**Outcome:** JPEG image showing preprocessed and augmented training frames.

---

## 11. Perform Batch Inference on Test Set

**Objective:** Run detection across the entire test split and save annotated outputs.
**Why:** Evaluates model generalization on unseen data, generates qualitative results for review.
**Outcome:** Annotated images stored under `runs/detect/predict` with confidence overlays.

---

## 12. Single-Image Inference Example

**Objective:** Demonstrate model versatility by detecting helmets in a custom image.
**Why:** Shows real-world applicability, supports multiple image formats.
**Outcome:** Inline display of detection results, annotated with bounding boxes and confidence scores.

---

## 13. Video Stream Processing

**Objective:** Apply the trained model to an input video file frame by frame.
**Why:** Enables automated post-processing for recorded footage, supports archival analysis.
**Outcome:** Output video saved with per-frame annotation in AVI format.

---

## 14. Real‑Time Webcam Deployment

**Objective:** Stream live video from a webcam to the YOLO model for on‑the‑fly detection.
**Why:** Demonstrates end-to-end deployment, supports interactive safety monitoring applications.
**Outcome:** Live display window with bounding boxes, running at near real-time speeds.

---
