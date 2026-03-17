# 🤖 Stylized Diffusion Policies via VLM Data Alchemy

![System Architecture](plots/flowchart.png)

This repository contains the official Python implementation of **"Generating Personality: A Framework for Unsupervised Style Discovery in Robotic Manipulation."** The goal of this project is to train conditional diffusion policies capable of performing the Push-T task with multiple distinct behavioral styles (e.g., Aggressive, Gentle, Hesitant, Neutral). The approach leverages Vision-Language Models (VLMs) for automated dataset labeling and Classifier-Free Guidance (CFG) for controllable behavior generation.

## ✨ Key Features
* **Data Alchemy (VLM Auto-Labeling)**: Uses Gemini 2.5 Flash with Chain-of-Thought prompting to extract zero-shot kinematic styles (Aggressive, Gentle, Hesitant) from unannotated demonstrations.
* **Edge-Optimized Diffusion**: A lightweight 1D CNN U-Net architecture conditioned via Feature-wise Linear Modulation (FiLM), fully trainable on a MacBook Pro.
* **Classifier-Free Guidance (CFG)**: A formalized "style knob" to dynamically extrapolate and amplify behavioral intensity during inference.
* **Failure Analysis on Visuomotor Policies**: Extensive documentation on the *Spatial Aliasing Paradox* and the critical necessity of proprioceptive state-conditioning in continuous control.

## 📊 Experimental Results & The "Closed-Loop Paradox"
We provide detailed logs and visualizations showing that while our stylized diffusion policy achieves excellent geometric differentiation in **open-loop** settings, vision-only architectures suffer from catastrophic covariate shift in **closed-loop** environments. 
Read the [Full Paper (PDF)](link-to-your-ECE285_Final.pdf) for an in-depth theoretical analysis of why CNN translation-invariance necessitates absolute coordinate anchoring.

## Project Structure & Pipeline
Our implementation is divided into several main stages, each mapping to a corresponding Python script:

### Stage 1: VLM Auto-Labeling (Data Alchemy)
**File**: [`process_full_dataset.py`](./process_full_dataset.py)
Uses the Google Gemini 2.5 Flash API to parse continuous action sequence sliding-windows from the `lerobot/pusht` dataset. The VLM acts as an automated annotator, labeling movements into specific styles ("Aggressive", "Gentle", "Hesitant", "Neutral"). The resulting classifications and confidence scores are written sequentially to [`trajectory_styles.jsonl`](./trajectory_styles.jsonl).

### Stage 2: Conditional Diffusion Training
**File**: [`train_diffusion.py`](./train_diffusion.py)
Trains a conditional Diffusion Policy consisting of a 1D CNN U-Net. This model predicts action trajectories based on:
- **Visual Features**: Through a pre-trained ResNet-18 backbone.
- **State Information**: Processed by an MLP embedding.
- **Style Labels**: Mapped via an embedding layer.
- **Diffusion Time**: Sinusoidal positional embeddings.
  
The training implements a 10% dropout probability for the style condition to support Classifier-Free Guidance (CFG), saving checkpoints (both EMA and standard) to the `checkpoints/` directory.

### Stage 3: Classifier-Free Guidance (CFG) Inference
**File**: [`evaluate_cfg.py`](./evaluate_cfg.py)
Evaluates the trained Diffusion Policy offline by utilizing the CFG formulas. It visually and mathematically compares kinematic differences—such as velocity and jerk—between trajectories generated with different conditioned styles (e.g., "Aggressive" vs. "Gentle") at varying CFG weights (`w`).

### Stage 3.5: Closed-Loop Simulation Evaluation
**File**: [`evaluate_env.py`](./evaluate_env.py)
Runs the trained model within the `gym_pusht` simulation environment for closed-loop, physical state-based evaluations. It utilizes an action chunking strategy (k=8) to ensure smooth control horizons and calculates precise empirical metrics including the Task Success Rate and exact Kinematic Distributions (Velocity, Jerk) for each behavioral style.

### Stage 4: Asset Generation
**File**: [`generate_presentation_assets.py`](./generate_presentation_assets.py)
Generates necessary presentation assets, including kinematic Kernel Density Estimation (KDE) plot distributions, spatial plot comparisons depicting "Aggressive" and "Hesitant" paths, and animated trajectory GIFs. Output is saved directly to the `plots/` directory.

## Usage

### Prerequisites
- Python 3.9+
- [PyTorch](https://pytorch.org/) properly configured for your hardware (the default is configured for Apple Silicon `mps`).
- Associated dependencies: `torchvision`, `gymnasium`, `gym_pusht`, `lerobot`, `seaborn`, `google-generativeai`.

### To run the pipeline end-to-end:
1. **Data Parsing**: Run `python process_full_dataset.py`. Requires `export GEMINI_API_KEY="your-api-key"`.
2. **Policy Training**: Execute `python train_diffusion.py`. 
3. **CFG Testing**: Run `python evaluate_cfg.py` to test kinematic scaling.
4. **Environment Evaluation**: View closed loop results via `python evaluate_env.py`.
5. **Report Assets**: Generate visual plots with `python generate_presentation_assets.py`.
