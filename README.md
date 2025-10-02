# Training Autonomous Car Using Deep Learning

This repository contains the work for our APS360 project at the **University of Toronto**, where we designed, implemented, and tested a multimodal deep learning system for autonomous driving using the CARLA simulator.  

Our system predicts **steering angle**, **vehicle velocity**, and **traffic light state** from visual and sensor inputs, integrating them into a controller for safe navigation in simulated urban environments.

---

## Table of Contents
- [Motivation](#motivation)
- [System Overview](#system-overview)
- [Data Collection & Processing](#data-collection--processing)
- [Architecture](#architecture)
- [Baseline Model](#baseline-model)
- [Results](#results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Project Difficulty](#project-difficulty)
- [Contributors](#contributors)

---

## Motivation
Traffic accidents remain a leading cause of death in urban areas. In Toronto alone, **298 serious/fatal collisions** were recorded in 2023. This project explores the potential of deep learning to reduce human error by enabling cars to perceive the environment and make rapid driving decisions.

Traditional rule-based systems often fail in edge cases. By leveraging **CNN-based multimodal learning**, our system adapts to diverse real-world driving scenarios.

---

## System Overview
The car’s decision pipeline is composed of three main modules:
- **Steering Classifier** – predicts steering direction from images + turn signals  
- **Velocity Regressor** – predicts vehicle speed  
- **Traffic Light Classifier** – detects and classifies intersection lights  

*Insert Figure 1 from report here* – Overall illustration of the system pipeline.

---

## Data Collection & Processing
We used the **CARLA simulator** to generate five datasets:
- **Steering dataset:** images, steering angles, turn signals  
- **Velocity dataset:** images, velocities  
- **Traffic Light dataset:** cropped images of traffic signals, state labels  
- **Manual driving dataset:** images + all sensor data  
- **Test dataset:** unseen maps to evaluate generalization  

All images were resized to **224×224×3** for use with **ResNet-18**.  
Augmentation (brightness, contrast, flips, cutouts) was applied to traffic light data to improve generalization.

*Insert sample preprocessed images (Figure 2, Figure 3 from report)*

---

## Architecture
We use a **shared ResNet-18 backbone** with three task-specific heads:

- **Steering:** concatenates ResNet features with a turn-signal embedding → classifies 15 angle bins  
- **Velocity:** two fully connected layers → continuous regression output  
- **Traffic Light:** fully connected layers → classifies Red, Green, or No-Light  

📌 *Insert Figure 4 (Final Architecture Low Level Diagram)*  

| Module              | Accuracy / Error |
|---------------------|------------------|
| Steering Classifier | 83.9% (train), 53.1% (test) |
| Traffic Light       | 99.9% (train), 99.8% (test) |
| Velocity Regressor  | MAE = 0.96 m/s, R² = 0.85 |

---

## Baseline Model
We first built a **Ridge Regression baseline** using grayscale images + turn signals.  
- Flattened **160×120 images** into feature vectors  
- Regularization parameter α tuned manually  
- Achieved reasonable steering predictions but struggled with generalization  

📌 *Insert Figure 7 (Learning Curve) and Figure 8 (Demo in CARLA)*

---

## Results
### Quantitative
- **Steering Classifier:** 53.1% test accuracy, CE Loss = 1.79  
- **Traffic Light Classifier:** 99.8% test accuracy, CE Loss = 0.0457  
- **Velocity Regressor:** MAE = 0.96 m/s, MSE = 1.75, R² = 85.2%  

### Qualitative
- Performed well across most CARLA maps  
- Struggled with sharp turns and highway overfitting  
- Strong multimodal fusion helped capture context  

📌 *Insert result tables and CARLA driving demo screenshots (Figures 5 & 8)*

---

## Discussion
- Steering is the hardest task due to ambiguity and high variability  
- Traffic light classification was highly robust (>99%)  
- Velocity predictions were consistent, but influenced by controller dynamics  
- Multimodal design (image + signal input) improved performance significantly  

---

## Ethical Considerations
- Over-reliance on autonomous systems may reduce driver readiness  
- Potential misuse if modified for unsafe driving  
- Our models were tested mainly in **urban maps** – performance in rural settings remains uncertain  

---

## Project Difficulty
- Required integrating **three different models** into one system  
- Collecting balanced datasets for traffic lights and steering was especially challenging  
- Designing **multimodal fusion** (visual + sensor data) added complexity  
- Despite this, we achieved robust performance across tasks and learned valuable lessons in deep learning for autonomy  

---

## Contributors
- **Rudra Dey** – Model development, CARLA integration, report writing  
- **Pravin Kalaivannan** – Data collection, preprocessing, report sections  
- **Aadavan Vasudevan** – Velocity regression model, code optimization  
- **Abishan Baheerathan** – Data collection scripts, final report contributions  

---
