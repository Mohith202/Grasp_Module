# CNN-Based Grasp Detection with Attention Mechanism on GraspNet Dataset

## Abstract
This project presents a hybrid approach for robotic grasp detection by combining **CNN-based image segmentation** and a **simple attention mechanism** for robust feature extraction. These modules were evaluated in conjunction with the **GraspNet baseline architecture**. The final model demonstrated an improvement in **Average Precision (AP)**, reaching **55%**, by effectively identifying graspable regions on seen objects.

---

## Methodology

### 1. Image Segmentation using CNN
We developed a lightweight CNN architecture to segment images and isolate graspable objects.  
The segmentation mask reduces the search space for the grasp detection module.

### 2. Attention Mechanism
An attention block was introduced to refine grasp detection by focusing on high-importance regions.  
The module used **softmax-weighted activations** to highlight potential grasp points from fused CNN features and **PointNet++** features.

### 3. Integration with GraspNet Baseline
The modified architecture was integrated into the **GraspNet-baseline pipeline**.  
We experimented with:
- CNN-only  
- Attention  
- CNN + Attention  

Each configuration was trained and evaluated on standard GraspNet benchmarks.

---

## Experiments
We trained and evaluated all model configurations on the **GraspNet dataset**, using **Average Precision (AP)** as the primary metric.

**Ablation study:**
- CNN-only (Final Model)  
- Attention  
- CNN + Attention  

---

## Results

| Configuration   | Average Precision (AP) |
|-----------------|-------------------------|
| Attention       | 35%                    |
| CNN + Attention | 31%                    |
| CNN             | **55%**                |

---

## Conclusion
The integration of **CNN-based segmentation**, **Vision Mamba feature extraction**, and **attention mechanisms** improves grasp detection performance on the GraspNet dataset.  

This multi-module pipeline enables the model to focus on grasp-relevant regions, resulting in better accuracy.  

**Future Work:**
- Extend the approach to **real-time robotic systems**  
- Evaluate performance on **physical hardware**  

---
