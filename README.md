
## Denoising Diffusion Implicit Models for Unsupervised Hypertension Monitoring from Photoplethysmography Signals

>**This is the official implementation of the paper with the title “**Denoising Diffusion Implicit Models for Unsupervised Hypertension Monitoring from Photoplethysmography Signals**” by **Myung-Kyu Yi**, **Jongshill Lee**, **Jeyeon Lee**, and **In Young Kim** 
- 📄 https://
- 🔗 DOI : 

---

## 🧠 Paper Overview

The model leverages **Denoising Diffusion Implicit Models (DDIM)** to learn the distribution of normal PPG signals, enabling label-free detection of morphological anomalies indicative of hypertension.

**Abstract**: 
While blood pressure monitoring is essential for managing hypertension, obtaining labeled data from hypertensive patients remains challenging, limiting the use of supervised learning approaches. To overcome this limitation, this study presents an unsupervised hypertension detection framework based on Denoising Diffusion Implicit Models (DDIM) with photoplethysmography (PPG) signals. The model is trained exclusively on PPG signals collected from normotensive individuals to learn the distribution of normal physiological waveforms, enabling the detection of hypertensive patterns as deviations from the training distribution. A temporally conditioned 1D U-Net is used to estimate noise in the reverse diffusion process, enabling deterministic signal reconstruction within the DDIM framework. Experiments conducted on three publicly available datasets—BP Assessment, PPG-BP, and PulseDB—demonstrate high F1 scores of 0.9067, 0.9130, and 0.9481, respectively. These findings demonstrate the strong generalizability of the proposed framework across diverse clinical environments and dataset scales, underscoring its potential for real-world hypertension screening using label-efficient and diffusion-based modeling. 

Key contributions include:
- A 1D-DDIM framework tailored for physiological signals
- Customized 1D U-Net backbone with attention and timestep embeddings
- Evaluation across 3 public datasets: **BP Assessment**, **PPG-BP**, and **PulseDB**

---

## 📂 Dataset & Preprocessing

The framework was validated on three public datasets:

- **BP Assessment Dataset** is available at https://github.com/sanvsquezsz/PPG-based-BP-assessment
- **PPG-BP Dataset** is available at https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299
- **PulseDB** is available at https://github.com/pulselabteam/PulseDB
---

## 🧱 Model Architecture

The model follows a 1D UNet-based DDIM pipeline:

- **Forward Process**: Gaussian noise added over time steps  
- **Reverse Process**: Deterministic DDIM sampling with learned denoiser  
- **Backbone**: 1D U-Net with Timestep Embedding, Cross-Attention, and Residual Blocks  

Threshold $\tau$ is selected based on reconstruction error distribution of normal validation samples.

---

## 🧪 Results Summary

| Dataset         | Precision | Recall | F1 Score | AUC   |
|------------------|-----------|--------|----------|--------|
| BP Assessment    | 0.8947    | 0.9189 | 0.9067   | 0.9401 |
| PPG-BP           | 0.8400    | 1.0000 | 0.9130   | 0.9256 |
| PulseDB          | 0.9583    | 0.9388 | 0.9481   | 0.9785 |

Performance is reported using optimal threshold $\tau$ derived from normal validation data.

---

## 💻 Repository Structure

| File/Folder               | Description |
|---------------------------|-------------|
| `model.py`                | UNet-based 1D DDIM architecture |
| `train_ddim.py`           | Training loop (unsupervised) |
| `inference.py`            | Anomaly scoring & threshold-based detection |

---

## Citing This Repository

If our project is helpful for your research, please consider citing :

```
@article{yi2025xxx,
  title={Denoising Diffusion Implicit Models for Unsupervised Hypertension Monitoring from Photoplethysmography Signals},
  author={Myung-Kyu Yi, Jongshill Lee, Jeyeon Lee, and In Young Kim },
  journal={},
  volume={},
  Issue={},
  pages={},
  year={}
  publisher={}
}

```

## Contact

Please feel free to contact via email (<kainos14@hanyang.ac.kr>) if you have further questions.

