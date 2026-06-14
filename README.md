# AI-Based Bubble Segmentation for Automated Detection of Diffuse Gas Leaks

Bachelor project developed in collaboration with Gassco AS as part of the Bachelor of Engineering in Computer Engineering at Western Norway University of Applied Sciences (HVL).

## Project Overview

Industrial operators use leak detection spray to visualize diffuse gas leaks through bubble formation. Today, the assessment of these bubbles is largely manual and relies on subjective measurements.

This project investigates whether modern AI-based image segmentation techniques can automatically identify and segment transparent bubbles in video recordings, providing a foundation for future automated gas leak analysis.

The solution combines automatic annotation using Segment Anything Model 3 (SAM3) with a custom-trained U-Net segmentation model for pixel-level bubble detection.

## Problem

Small errors in estimating bubble size can lead to large errors when calculating leak volume and emission rates.

The challenge is particularly difficult because bubbles are:

- Transparent or semi-transparent
- Deformable
- Overlapping
- Sensitive to lighting conditions
- Embedded in noisy industrial environments

## Solution

The project pipeline consists of:

1. Video acquisition
2. Frame extraction
3. Automatic annotation using SAM3
4. Data preprocessing and augmentation
5. U-Net training
6. Semantic segmentation
7. Quantitative and qualitative evaluation

## Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- Segment Anything Model 3 (SAM3)
- Docker
- NumPy
- Matplotlib

## Technical Highlights

- Built a complete computer vision pipeline from data collection to model evaluation
- Automated image annotation using SAM3
- Developed and trained a U-Net segmentation model
- Created a custom laboratory environment for dataset generation
- Evaluated model robustness across controlled and industrial environments

## Dataset

Three datasets were used:

- Laboratory recordings using a controlled setup
- Bubble gun recordings
- Industrial recordings provided by Gassco AS

The laboratory environment included controlled lighting, fixed camera positioning, and greenscreen backgrounds to create high-quality training data.

## Results

The model achieved promising segmentation performance under controlled conditions and demonstrated the feasibility of AI-based bubble segmentation.

Key findings:

- Successful segmentation of transparent bubbles
- Effective use of SAM3 for automated annotation
- Strong performance on laboratory data
- Limited generalization to industrial environments due to domain shift and environmental complexity

The project therefore demonstrates a successful proof-of-concept for future automated leak detection systems.

## Repository Structure
