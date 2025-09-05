Pull-Up Detection System
A computer vision and machine learning system that automatically detects and analyzes pull-up exercises using pose estimation and neural networks.
## Overview
This project combines Google's MediaPipe pose estimation with recurrent neural networks to provide real-time pull-up form analysis and repetition counting. The system can detect exercise phases, count repetitions accurately, and identify form issues to help users improve their workout technique.
## Features

## Real-time Pull-Up Detection: Automatically counts pull-up repetitions in video input
Form Analysis: Identifies lapses in form and provides feedback on exercise technique
Phase Detection: Recognizes different stages of the pull-up movement (hang, pull, peak, descent)
REST API: Easy integration with web applications through HTTP endpoints
Transfer Learning: Leverages MediaPipe's pre-trained models for robust pose estimation

# Architecture
## Object Detection Module (objdetect.py)

Utilizes Google MediaPipe API for pose estimation
Analyzes body landmarks (arms, wrists, shoulders, etc.)
Defines threshold-based detection for pull-up phases
Provides labeled training data for the neural network
Handles video capture and frame analysis

Key Functions:

Angle calculation between body joints
Frame-by-frame pose analysis
Video capture management
Exercise phase classification

## Neural Network Module (rnn.py)

Implements a Recurrent Neural Network for sequence recognition
Processes pose estimation data to detect patterns in pull-up form
Uses transfer learning with MediaPipe for feature extraction
Focuses on temporal sequence analysis for improved accuracy
Classifies exercise quality and counts repetitions

Features:

Pattern recognition in pull-up sequences
Form quality assessment
Repetition counting validation
Temporal analysis of movement patterns

## REST API

Provides HTTP endpoints for web application integration
Supports standard REST operations (GET, POST, PUT, DELETE)
Enables frontend applications to access backend functionality
Decouples the web interface from the core detection system

## Installation
Prerequisites

Python 3.8+
OpenCV
MediaPipe
TensorFlow/PyTorch (for RNN implementation)
NumPy
Flask/FastAPI (for REST API)
