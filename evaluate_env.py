"""
evaluate_env.py
Stage 3.5: Closed-Loop Simulation Evaluation & KDE Kinematics

Loads the trained Diffusion Policy and evaluates it rigorously in the 
LeRobot PushT environment. Uses Action Chunking (k=8) and runs 
20 full episodes per condition ("Aggressive", "Gentle", "Hesitant") at w=2.0
to calculate exact Task Success Rates and exact Kinematic distributions.
"""

import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import gymnasium as gym
# We import lerobot to ensure the envs are registered in gymnasium
import lerobot.envs 
import gym_pusht

from train_diffusion import DiffPolicy, cosine_beta_schedule, StylizedPushTDataset
from evaluate_cfg import DDPM_CFG, compute_kinematics

# ----------------- #
# 1. Closed-Loop Evaluation Rollout
# ----------------- #
def run_eval_episodes(env, model_ddpm, style_idx, num_episodes=20, chunk_size=8, w=2.0, action_min=None, action_max=None, state_min=None, state_max=None, device="mps"):
    """
    Runs `num_episodes` closed-loop rollouts for a single stylistic condition.
    Returns:
       - success_rate (float percentage)
       - all_velocities (list of floats)
       - all_jerks (list of floats)
    """
    successes = 0
    all_velocities = []
    all_jerks = []
    
    # We use a fixed evaluation seed for consistent initial block placement across styles
    base_seed = 10000 
    
    print(f"\n--- Testing Condition: Style Index {style_idx.item()} (w={w}) ---")
    
    for ep in tqdm(range(num_episodes), desc="Episodes", leave=False):
        obs, info = env.reset(seed=base_seed + ep)
        done = False
        truncated = False
        step_count = 0
        
        # Max steps as safety net (Push-T standard is usually ~300)
        max_steps = 300
        
        ep_executed_actions = []
        
        while not done and not truncated and step_count < max_steps:
            # 1. Extract and Format Observation
            if isinstance(obs, dict):
                img = obs.get("pixels", obs)
                state = obs.get("agent_pos", None)
            else:
                img = obs
                state = None
                
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img, dtype=torch.float32)
                
            # If [H, W, C] shape, convert to [C, H, W]
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
                
            # Scale uint8 [0, 255] to float [0, 1]
            if img.max() > 1.0:
                img = img / 255.0
                
            img_batch = img.unsqueeze(0).to(device)
            
            # Extract and format State
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            state_batch = state.unsqueeze(0).to(device)
            state_batch = 2.0 * (state_batch - state_min) / (state_max - state_min) - 1.0
            
            # 2. Diffusion Inference (Generate 16 steps)
            # We don't use fixed_noise here because closed-loop control relies on new sampling
            with torch.no_grad():
                pred_action_seq = model_ddpm.sample(img_batch, state_batch, style_idx, shape=(1, 16, 2), w=w)
            
            # 3. Unnormalize
            pred_action_seq = (pred_action_seq + 1.0) / 2.0 * (action_max - action_min) + action_min
            pred_action_seq = pred_action_seq.squeeze(0).cpu().numpy() # [16, 2]
            
            # 4. Action Chunking Execution
            # Execute only the first `chunk_size` actions to allow rapid control horizon updates
            chunk = pred_action_seq[:chunk_size]
            for act in chunk:
                # Env step expects dict or array depending on lerobot version wrap
                # Default gym from lerobot usually expects standard Box array
                obs, reward, done, truncated, info = env.step(act)
                ep_executed_actions.append(act)
                step_count += 1
                
                if done or truncated or step_count >= max_steps:
                    break
        
        # Check success
        # In Push-T, hitting `done=True` before timeout essentially means task success
        # Or alternatively tracking `info.get('is_success')`. we use standard done.
        is_success = info.get('is_success', done) 
        if is_success:
            successes += 1
            
        # Store Kinematics for the *executed* actions
        if len(ep_executed_actions) > 3:
            ep_traj = np.array(ep_executed_actions)[np.newaxis, ...] # [1, L, 2]
            v, j = compute_kinematics(torch.tensor(ep_traj))
            all_velocities.append(v)
            all_jerks.append(j)

    success_rate = (successes / num_episodes) * 100.0
    print(f"  -> Success Rate: {success_rate:.1f}% ({successes}/{num_episodes})")
    
    return success_rate, all_velocities, all_jerks

# ----------------- #
# 2. Main Script
# ----------------- #
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Starting Stage 3.5: Closed-Loop Evaluation on {device} ---")
    
    ckpt_path = "checkpoints/policy_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Ensure training is running/completed.")
        return

    # Load Model
    policy = DiffPolicy().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    ddpm_sampler = DDPM_CFG(policy, timesteps=100, device=device)
    
    # Needs dataset solely to retrieve the specific min/max tracking bounds
    dataset = StylizedPushTDataset(seq_len=16)
    action_min = dataset.action_min.to(device)
    action_max = dataset.action_max.to(device)
    state_min = dataset.state_min.to(device)
    state_max = dataset.state_max.to(device)
    
    # Ensure plots dir exists
    os.makedirs("plots", exist_ok=True)
    
    # Make PushT Env 
    # Must use dict observation space to match lerobot pixel extraction
    # Standard format: obs_type="pixels" 
    try:
        env = gym.make("lerobot/pusht", obs_type="pixels_agent_pos", render_mode="rgb_array")
    except Exception as e:
        print("Fallback trying 'gym_pusht/PushT-v0' directly:", e)
        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    # Mapping: {"Aggressive": 0, "Gentle": 1, "Hesitant": 2, "Neutral": 3}
    conditions = {
        "Aggressive": 0,
        "Gentle": 1,
        "Hesitant": 2
    }
    
    # Config
    NUM_EPISODES = 20
    CHUNK_SIZE = 8
    CFG_WEIGHT = 2.0
    
    metrics = {}
    
    # Run w=0.0 first to check closed-loop base policy
    test_weights = [0.0, CFG_WEIGHT]
    
    for w in test_weights:
        print(f"\n================ Eval w={w} ================")
        for name, s_idx in conditions.items():
            style_tensor = torch.tensor([s_idx], device=device, dtype=torch.long)
            sr, veles, jerks = run_eval_episodes(
                env, 
                ddpm_sampler, 
                style_idx=style_tensor, 
                num_episodes=NUM_EPISODES, 
                chunk_size=CHUNK_SIZE, 
                w=w, 
                action_min=action_min, 
                action_max=action_max,
                state_min=state_min,
                state_max=state_max,
                device=device
            )
            # Only store metrics for w=2.0 for plotting
            if w == CFG_WEIGHT:
                metrics[name] = {"sr": sr, "vel": veles, "jerk": jerks}
        
    env.close()

    # ----------------- #
    # 3. Plotting KDE
    # ----------------- #
    print("\n--- Generating KDE Distribution Plots ---")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {"Aggressive": "red", "Gentle": "blue", "Hesitant": "green"}
    
    # Velocity Plot
    ax_vel = axes[0]
    for name in conditions.keys():
        if len(metrics[name]["vel"]) > 0:
            sns.kdeplot(metrics[name]["vel"], bw_adjust=0.5, fill=True, alpha=0.3, 
                        color=colors[name], label=f"{name} (SR: {metrics[name]['sr']:.0f}%)", ax=ax_vel)
    ax_vel.set_title("Distribution of Episode Mean Velocity")
    ax_vel.set_xlabel("Mean Velocity (||v||)")
    ax_vel.set_ylabel("Density")
    ax_vel.legend()

    # Jerk Plot
    ax_jerk = axes[1]
    for name in conditions.keys():
        if len(metrics[name]["jerk"]) > 0:
            sns.kdeplot(metrics[name]["jerk"], bw_adjust=0.5, fill=True, alpha=0.3, 
                        color=colors[name], label=name, ax=ax_jerk)
    ax_jerk.set_title("Distribution of Episode Mean Jerk (Smoothness Penalty)")
    ax_jerk.set_xlabel("Mean Jerk")
    ax_jerk.set_ylabel("Density")
    ax_jerk.legend()

    plt.suptitle(f"Closed-Loop Kinematics Analysis | Push-T Environment (CFG w={CFG_WEIGHT})", fontsize=16)
    plt.tight_layout()
    
    plot_path = "plots/kinematic_kde_results.png"
    plt.savefig(plot_path, dpi=300)
    print(f"KDE Plot successfully saved to: {plot_path}")

if __name__ == "__main__":
    main()
