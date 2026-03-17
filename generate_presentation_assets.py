import os
import json
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import torchvision.transforms.functional as TF
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def create_distribution_chart(styles, out_dir):
    counts = {}
    for style in ["Aggressive", "Gentle", "Hesitant", "Neutral"]:
        counts[style] = styles.count(style)
        
    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.keys(), counts.values(), color=['salmon', 'lightgreen', 'lightblue', 'lightgray'])
    plt.title('VLM Labelled Trajectory Style Distribution', fontsize=16)
    plt.ylabel('Number of Episodes', fontsize=14)
    plt.xlabel('Style Categories', fontsize=14)
    
    total = sum(counts.values())
    
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / total) * 100 if total > 0 else 0
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f"{int(yval)}\n({pct:.1f}%)", 
                 va='bottom', ha='center', fontsize=12)
        
    plt.tight_layout()
    plt.ylim(0, max(counts.values()) * 1.15)
    out_path = os.path.join(out_dir, 'style_distribution.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def create_kinematic_plot(episode_styles, ds, out_dir):
    print("Computing kinematic densities... (This may take a minute)")
    
    style_episodes = defaultdict(list)
    for ep_idx, style in episode_styles.items():
        if style in ["Aggressive", "Gentle", "Hesitant"]:
            style_episodes[style].append(ep_idx)
            
    velocities = defaultdict(list)
    fps = float(ds.meta.fps) if hasattr(ds.meta, 'fps') else 10.0
    dt = 1.0 / fps
    
    for style, ep_indices in style_episodes.items():
        sample_indices = list(ep_indices)[:50]
        
        for ep_idx in sample_indices:
            ep = next((e for e in ds.meta.episodes if e['episode_index'] == ep_idx), None)
            if ep is None: continue
            
            start_idx = ep['dataset_from_index']
            end_idx = ep['dataset_to_index']
            
            actions = [ds.hf_dataset[i]['action'] for i in range(start_idx, end_idx)]
            actions = torch.stack([torch.tensor(a) if not isinstance(a, torch.Tensor) else a for a in actions]).float()
            
            diffs = torch.diff(actions, dim=0)
            speeds = torch.norm(diffs, dim=1) / dt
            mean_speed = speeds.mean().item()
            velocities[style].append(mean_speed)
            
    plt.figure(figsize=(10, 6))
    colors = {"Aggressive": "red", "Gentle": "green", "Hesitant": "blue"}
    
    if HAS_SEABORN:
        for style, vels in velocities.items():
            if len(vels) > 1:
                sns.kdeplot(vels, label=style, color=colors[style], fill=True, alpha=0.3, linewidth=2)
    else:
        print("Note: Seaborn not found. Running with basic Matplotlib fallback. Install seaborn for better KDE plots.")
        for style, vels in velocities.items():
            if len(vels) > 1:
                plt.hist(vels, bins=10, alpha=0.5, label=style, density=True, color=colors[style])
                
    plt.title('Kinematic Analysis: Mean Velocity Distribution by Style', fontsize=16)
    plt.xlabel('Mean Velocity (Pixels / Second)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'kinematic_densities.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def create_gifs(agg_idx, hes_idx, ds, out_dir):
    print("Extracting highest confidence trajectories for GIFs...")
            
    def save_gif(ep_idx, filename, fps):
        if ep_idx is None:
            print(f"Skipping {filename}, no valid episode index found.")
            return
            
        print(f"Generating {filename} from episode {ep_idx}...")
        ep = next((e for e in ds.meta.episodes if e['episode_index'] == ep_idx), None)
        if ep is None: return
        
        start_idx = ep['dataset_from_index']
        end_idx = ep['dataset_to_index']
        
        images = []
        for i in range(start_idx, end_idx):
            try:
                img_data = ds.hf_dataset[i]['observation.image']
                if hasattr(img_data, 'size') and hasattr(img_data, 'mode'):
                    images.append(img_data)
                else:
                    raise TypeError("Not a direct PIL Image")
            except Exception:
                obs_tensor = ds[i]['observation.image']
                if obs_tensor.min() < 0:
                    obs_tensor = (obs_tensor + 1.0) / 2.0
                obs_tensor = torch.clamp(obs_tensor, 0.0, 1.0)
                img = TF.to_pil_image(obs_tensor)
                images.append(img)
            
        if len(images) > 0:
            duration = int(1000 / fps)
            out_path = os.path.join(out_dir, filename)
            images[0].save(out_path, save_all=True, append_images=images[1:], loop=0, duration=duration)
            print(f"Saved {out_path} with {len(images)} frames.")

    fps = float(ds.meta.fps) if hasattr(ds.meta, 'fps') else 10.0
    save_gif(agg_idx, 'aggressive_demo.gif', fps)
    save_gif(hes_idx, 'hesitant_demo.gif', fps)

def create_trajectory_plot(agg_idx, agg_conf, hes_idx, hes_conf, ds, out_dir):
    print("Extracting and plotting spatial trajectories...")
    
    agg_actions = []
    hes_actions = []
    agg_time = 0.0
    hes_time = 0.0
    
    for ep in ds.meta.episodes:
        if ep['episode_index'] == agg_idx:
            start_idx = ep['dataset_from_index']
            end_idx = ep['dataset_to_index']
            agg_time = ep['length'] / ds.meta.fps
            agg_actions = [ds.hf_dataset[i]['action'] for i in range(start_idx, end_idx)]
        elif ep['episode_index'] == hes_idx:
            start_idx = ep['dataset_from_index']
            end_idx = ep['dataset_to_index']
            hes_time = ep['length'] / ds.meta.fps
            hes_actions = [ds.hf_dataset[i]['action'] for i in range(start_idx, end_idx)]
            
    agg_actions = torch.stack([torch.tensor(a) if not isinstance(a, torch.Tensor) else a for a in agg_actions])
    hes_actions = torch.stack([torch.tensor(a) if not isinstance(a, torch.Tensor) else a for a in hes_actions])
    
    # Calculate Mean Speeds
    dt = 1.0 / ds.meta.fps
    agg_speeds = torch.norm(torch.diff(agg_actions.float(), dim=0), dim=1) / dt
    hes_speeds = torch.norm(torch.diff(hes_actions.float(), dim=0), dim=1) / dt
    
    agg_mean_speed = agg_speeds.mean().item()
    hes_mean_speed = hes_speeds.mean().item()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left subplot: Aggressive
    ax = axes[0]
    ax.plot(agg_actions[:, 0].numpy(), agg_actions[:, 1].numpy(), color='red', label='Path')
    ax.plot(agg_actions[0, 0].numpy(), agg_actions[0, 1].numpy(), 'g*', markersize=15, label='Start')
    ax.plot(agg_actions[-1, 0].numpy(), agg_actions[-1, 1].numpy(), 'ks', markersize=10, label='End')
    ax.set_title(f"Style: Aggressive |  Time: {agg_time:.1f}s  |  Speed: {agg_mean_speed:.1f} px/s")
    ax.set_xlabel('X Coordinate (Pixels)')
    ax.set_ylabel('Y Coordinate (Pixels)')
    ax.text(0.5, -0.15, "Less directional changes, faster path to target.", transform=ax.transAxes, ha='center', va='top', wrap=True)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend()
    
    # Right subplot: Hesitant
    ax = axes[1]
    ax.plot(hes_actions[:, 0].numpy(), hes_actions[:, 1].numpy(), color='blue', label='Path')
    ax.plot(hes_actions[0, 0].numpy(), hes_actions[0, 1].numpy(), 'g*', markersize=15, label='Start')
    ax.plot(hes_actions[-1, 0].numpy(), hes_actions[-1, 1].numpy(), 'ks', markersize=10, label='End')
    ax.set_title(f"Style: Hesitant |  Time: {hes_time:.1f}s  |  Speed: {hes_mean_speed:.1f} px/s")
    ax.set_xlabel('X Coordinate (Pixels)')
    ax.set_ylabel('Y Coordinate (Pixels)')
    ax.text(0.5, -0.15, "More directional changes, and slower progress.", transform=ax.transAxes, ha='center', va='top', wrap=True)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    out_path = os.path.join(out_dir, "vlm_style_discovery.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def main():
    jsonl_path = "trajectory_styles.jsonl"
    out_dir = "plots"
    
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
        
    episode_styles = {}
    confidences = {}
    
    agg_idx, agg_conf = -1, -1.0
    hes_idx, hes_conf = -1, -1.0
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            ep_idx = data["episode_index"]
            style = data["final_style"]
            conf = data["confidence"]
            
            confidences[ep_idx] = conf
            if style != "Discarded":
                episode_styles[ep_idx] = style
                
            if style == "Aggressive" and conf > agg_conf:
                agg_conf = conf
                agg_idx = ep_idx
            elif style == "Hesitant" and conf > hes_conf:
                hes_conf = conf
                hes_idx = ep_idx
                
    create_distribution_chart(list(episode_styles.values()), out_dir)
    
    print("Loading dataset 'lerobot/pusht'...")
    ds = LeRobotDataset("lerobot/pusht")
    
    create_kinematic_plot(episode_styles, ds, out_dir)
    create_gifs(agg_idx, hes_idx, ds, out_dir)
    create_trajectory_plot(agg_idx, agg_conf, hes_idx, hes_conf, ds, out_dir)
    
    print(f"\nAll presentation assets have been saved to the '{out_dir}/' directory.")

if __name__ == "__main__":
    main()
