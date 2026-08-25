---
title: Road Damage Detection
emoji: 🛣️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Road Damage Detection

A computer vision application that detects road damage using a custom-trained YOLO model.

The application provides a web interface where users can upload a road image and receive an annotated image with detected damage areas, class names, confidence scores, and bounding boxes.

## Supported Damage Classes

- D00
- D01
- D10
- D11
- D20
- D40
- D43
- D44
- D50

## Project Structure

```text
.
├── app.py
├── best.pt
├── index.html
├── requirements.txt
└── README.md