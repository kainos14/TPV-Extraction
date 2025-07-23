
## A Topological Descriptor of Photoplethysmography Waveforms for Blood Pressure Profiling

>**This is the official implementation of the paper with the title “**A Topological Descriptor of Photoplethysmography Waveforms for Blood Pressure Profiling**” by **Myung-Kyu Yi**, **Jongshill Lee**, **Jeyeon Lee**, and **In Young Kim** 
- 📄 https://
- 🔗 DOI : 

---

## 🧠 Paper Overview

**Abstract**: 
Conventional PPG-based features often rely on localized waveform descriptors, making them highly sensitive to noise and inter-subject variability, which limits the accuracy and personalization of blood pressure monitoring. To address these issues, we propose the Topological Prior Vector (TPV), a 33-dimensional feature representation derived from persistent homology of PPG signals. Unlike traditional approaches, TPV compactly encodes the global topological structure of waveforms, providing interpretable and subject-specific characteristics. We evaluated TPV on normotensive and hypertensive cohorts using both short- and long-term recordings, assessing intra-subject stability, inter-subject separability, and correlations with systolic and diastolic blood pressure. Experimental results demonstrate that TPV exhibits strong temporal consistency and significant associations with blood pressure variability, supporting its potential as a non-invasive and physiologically meaningful descriptor for personalized blood pressure monitoring.

Key contributions include:
- We introduce TPV, a 33-dimensional topological representation of PPG and ECG signals, showing strong BP correlation, high intra-subject stability, and variability under hypertensive conditions for personalized BP monitoring.
- TPV demonstrates inter-subject separability and BP-related differences, enables accurate hypertension classification with simple models, and offers physiologically interpretable features validated through SHAP analysis.
- Evaluation across two public dataset: **BP Assessment** and **PulseDB**

---

## 📂 Dataset & Preprocessing

The framework was validated on three public datasets:

- **BP Assessment Dataset** is available at https://github.com/sanvsquezsz/PPG-based-BP-assessment
- **PulseDB** is available at https://github.com/pulselabteam/PulseDB
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
  title={A Topological Descriptor of Photoplethysmography Waveforms for Blood Pressure Profiling},
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

