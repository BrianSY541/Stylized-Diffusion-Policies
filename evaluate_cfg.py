"""
evaluate_cfg.py
Stage 3: Classifier-Free Guidance (CFG) Inference & Evaluation

This script loads the trained Diffusion Policy, takes a starting observation from 
the LeRobot Push-T dataset, and mathematically evaluates the kinematic differences 
between generated "Aggressive" and "Gentle" trajectories using various CFG weights.
"""

import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Import the architecture directly from the training script
# (Ensures 100% exact architectural match for the loaded weights)
from train_diffusion import StylizedPushTDataset, DiffPolicy, cosine_beta_schedule

# ----------------- #
# 1. CFG DDPM Inference Setup
# ----------------- #
class DDPM_CFG:
    """
    DDPM class strictly for Reverse Diffusion Sampling with Classifier-Free Guidance.
    """
    def __init__(self, model, timesteps=100, device="mps"):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Precompute DDPM step variables
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        
        # Precompute posterior mean coefficients for x0 clamping
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @torch.no_grad()
    def p_sample_cfg(self, x, obs, state, t, style_idx, w=1.5):
        """
        Executes a single DDPM reverse step using the CFG formula.
        math: eps_tilde = eps_uncond + w * (eps_cond - eps_uncond)
        """
        B = x.shape[0]
        t_batched = torch.full((B,), t, device=self.device, dtype=torch.long)
        
        # 1. Unconditioned Noise Prediction (Null Token is index 4)
        style_null = torch.full((B,), 4, device=self.device, dtype=torch.long)
        eps_uncond = self.model(x, obs, state, style_null, t_batched)
        
        # 2. Conditioned Noise Prediction
        eps_cond = self.model(x, obs, state, style_idx, t_batched)
        
        # 3. Classifier-Free Guidance Extrapolation
        eps_tilde = eps_uncond + w * (eps_cond - eps_uncond)
        
        # 4. Standard DDPM reverse step math to get x_{t-1} using clamped x0 prediction
        # Instead of directly using eps_tilde which explodes under CFG (w > 1), we predict x0 explicitly
        pred_x0 = (x - self.sqrt_one_minus_alphas_cumprod[t] * eps_tilde) / self.sqrt_alphas_cumprod[t]
        
        # CRITICAL: Clamp x0 to physical [-1, 1] bounds to mathematically prevent exponential extrapolation explosions
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute the model posterior mean
        model_mean = self.posterior_mean_coef1[t] * pred_x0 + self.posterior_mean_coef2[t] * x
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            posterior_var_t = self.posterior_variance[t]
            return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, obs, state, style_idx, shape, w=1.5, fixed_noise=None):
        """
        Generates a full trajectory starting from pure noise (t=T down to t=0).
        Accepts optional `fixed_noise` to ensure direct style comparison is valid.
        """
        if fixed_noise is not None:
            x = fixed_noise.clone()
            # CRITICAL: Ensure transition noises during the loop are perfectly synchronized across style comparisons
            torch.manual_seed(42)
        else:
            x = torch.randn(shape, device=self.device)
        
        for t in reversed(range(self.timesteps)):
            x = self.p_sample_cfg(x, obs, state, t, style_idx, w=w)
            
        return x

# ----------------- #
# 2. Kinematic Metrics
# ----------------- #
def compute_kinematics(trajectory):
    """
    Computes Mean Velocity and Jerk for a 2D action trajectory [seq_len, 2]
    trajectory shape: [1, 16, 2] -> numpy array
    """
    traj = trajectory.squeeze(0).cpu().numpy() # [16, 2]
    
    # Velocity (1st derivative): v = dx/dt
    velocities = np.diff(traj, axis=0) # [15, 2]
    speeds = np.linalg.norm(velocities, axis=1) # [15]
    mean_velocity = np.mean(speeds)
    
    # Acceleration (2nd derivative): a = dv/dt
    accelerations = np.diff(velocities, axis=0) # [14, 2]
    
    # Jerk (3rd derivative): j = da/dt. Represents smoothness/hesitation
    jerks = np.diff(accelerations, axis=0) # [13, 2]
    jerk_magnitudes = np.linalg.norm(jerks, axis=1) # [13]
    mean_jerk = np.mean(jerk_magnitudes)
    
    return mean_velocity, mean_jerk

# ----------------- #
# 3. Main Evaluation Script
# ----------------- #
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Starting Stage 3: Evaluation CFG on {device} ---")
    
    ckpt_path = "checkpoints/policy_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint '{ckpt_path}' not found.")
        print("Please ensure train_diffusion.py has finished its first 10 epochs or completed fully.")
        return
        
    print("Loading architecture and weights...")
    policy = DiffPolicy().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    ddpm_sampler = DDPM_CFG(policy, timesteps=100, device=device)
    
    # Load exactly 1 Observation from a test trajectory
    # Using seq_len=16 to ensure standard properties
    dataset = StylizedPushTDataset(seq_len=16)
    
    # Grab an arbitrary observation for testing (e.g. index 500)
    test_idx = 500 if len(dataset) > 500 else 0
    obs_tensor, state_tensor, gt_action, _ = dataset[test_idx]
    
    # Add batch dimension: [1, 3, H, W]
    obs_batch = obs_tensor.unsqueeze(0).to(device)
    state_batch = state_tensor.unsqueeze(0).to(device)
    
    # Style targets to compare
    # Map: {"Aggressive": 0, "Gentle": 1, "Hesitant": 2, "Neutral": 3}
    styles_to_test = {"Aggressive": 0, "Gentle": 1}
    cf_weights = [0.0, 1.5, 3.0]
    
    # Generate fixed initial noise to ensure direct, valid style comparison
    fixed_noise = torch.randn((1, 16, 2), device=device)
    
    # Store dataset action bounds for unnormalization
    action_min = dataset.action_min.to(device)
    action_max = dataset.action_max.to(device)
    
    results = {}
    
    print("\n--- Generating Trajectories via CFG ---")
    for style_name, style_int in styles_to_test.items():
        results[style_name] = {}
        style_idx = torch.tensor([style_int], device=device, dtype=torch.long)
        
        for w in cf_weights:
            print(f"Sampling '{style_name}' with CFG weight w={w:.1f}...")
            # shape: [B, seq_len, action_dim]
            gen_traj = ddpm_sampler.sample(obs_batch, state_batch, style_idx, shape=(1, 16, 2), w=w, fixed_noise=fixed_noise)
            
            # Compute Kinematics
            # mean_v, mean_j = compute_kinematics(gen_traj)

            # Unnormalize back to the original physical pixel bounds of Push-T
            # Formula: x_{unnorm} = (x_{norm} + 1.0) / 2.0 * (max - min) + min
            gen_traj = (gen_traj + 1.0) / 2.0 * (action_max - action_min) + action_min
            
            # Compute Kinematics
            mean_v, mean_j = compute_kinematics(gen_traj)

            results[style_name][w] = {
                "trajectory": gen_traj.squeeze(0).cpu().numpy(),
                "velocity": mean_v,
                "jerk": mean_j
            }
            print(f"  -> Mean Velocity: {mean_v:.4f} | Mean Jerk: {mean_j:.4f}")

    print("\n--- Plotting Trajectories ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Stylized Diffusion Policy: Classifier-Free Guidance", fontsize=16)
    
    for i, w in enumerate(cf_weights):
        ax = axes[i]
        
        # The Ground Truth trajectory retrieved from dataset __getitem__ was normalized along with the images.
        # It must ALSO be unnormalized for the visualization reference line to be accurate.
        gt_traj = gt_action.to(device)
        gt_traj = (gt_traj + 1.0) / 2.0 * (action_max - action_min) + action_min
        gt_traj = gt_traj.cpu().numpy()
        
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'k--', alpha=0.5, label='Ground Truth')
        
        # Plot generated styles
        agg_traj = results["Aggressive"][w]["trajectory"]
        gen_traj = results["Gentle"][w]["trajectory"]
        
        ax.plot(agg_traj[:, 0], agg_traj[:, 1], 'r-', linewidth=2, label=f'Aggressive')
        ax.plot(gen_traj[:, 0], gen_traj[:, 1], 'b-', linewidth=2, label=f'Gentle')
        
        # Mark start points
        ax.scatter(agg_traj[0, 0], agg_traj[0, 1], c='red', marker='o')
        ax.scatter(gen_traj[0, 0], gen_traj[0, 1], c='blue', marker='o')
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], c='black', marker='o', alpha=0.5)
        
        ax.set_title(f"CFG Weight: w = {w}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Print metrics to plot
        text_str = (f"w={w}\n"
                    f"Agg V: {results['Aggressive'][w]['velocity']:.3f} | J: {results['Aggressive'][w]['jerk']:.3f}\n"
                    f"Gen V: {results['Gentle'][w]['velocity']:.3f} | J: {results['Gentle'][w]['jerk']:.3f}")
        ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plot_path = "./plots/cfg_style_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nEvaluating complete! Visualization saved to: {plot_path}")

if __name__ == "__main__":
    main()
