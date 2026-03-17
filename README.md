# 🤖 Stylized Diffusion Policies via VLM Data Alchemy

![System Architecture](plots/flowchart.png) *(可放入您報告中的架構圖)*

This repository contains the official PyTorch implementation of **"Generating Personality: A Framework for Unsupervised Style Discovery in Robotic Manipulation."** It demonstrates how to decouple *what* a robot does from *how* it does it, leveraging Vision-Language Models (VLMs) as offline annotators and deploying lightweight diffusion policies on edge hardware (Apple M1).

## ✨ Key Features
* **Data Alchemy (VLM Auto-Labeling)**: Uses Gemini 2.5 Flash with Chain-of-Thought prompting to extract zero-shot kinematic styles (Aggressive, Gentle, Hesitant) from unannotated demonstrations.
* **Edge-Optimized Diffusion**: A lightweight 1D CNN U-Net architecture conditioned via Feature-wise Linear Modulation (FiLM), fully trainable on a MacBook Pro.
* **Classifier-Free Guidance (CFG)**: A formalized "style knob" to dynamically extrapolate and amplify behavioral intensity during inference.
* **Failure Analysis on Visuomotor Policies**: Extensive documentation on the *Spatial Aliasing Paradox* and the critical necessity of proprioceptive state-conditioning in continuous control.

## 📊 Experimental Results & The "Closed-Loop Paradox"
We provide detailed logs and visualizations showing that while our stylized diffusion policy achieves excellent geometric differentiation in **open-loop** settings, vision-only architectures suffer from catastrophic covariate shift in **closed-loop** environments. 
Read the [Full Paper (PDF)](link-to-your-ECE285_Final.pdf) for an in-depth theoretical analysis of why CNN translation-invariance necessitates absolute coordinate anchoring.

## 🚀 Quick Start
### 1. Environment Setup
```bash
python -m venv lerobot_env
source lerobot_env/bin/activate
pip install -r requirements.txt
```

### 2. Run Data Alchemy (Stage 1)
```bash
export GEMINI_API_KEY="your_api_key_here"
python process_full_dataset.py
```

### 3. Train the Diffusion Policy (Stage 2)
```bash
python train_diffusion.py
```

### 4. Evaluate & Visualize (Stage 3)
```bash
python evaluate_cfg.py  # Generates open-loop style comparisons
python evaluate_env.py  # Generates closed-loop KDE kinematic distributions
```
