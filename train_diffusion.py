"""
train_diffusion.py
Production Training Script for Stage 2: Conditional Diffusion Training (Stylized Diffusion Policies)
"""

import os
import json
import math
import copy
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ----------------- #
# 1. Dataset Integration & Rebalancing
# ----------------- #
class StylizedPushTDataset(Dataset):
    def __init__(self, jsonl_path="trajectory_styles.jsonl", seq_len=16):
        print("Loading LeRobotDataset 'lerobot/pusht'...")
        self.ds = LeRobotDataset("lerobot/pusht")
        self.seq_len = seq_len
        
        # Mapping styles to integers (index 4 is NULL)
        self.style_map = {"Aggressive": 0, "Gentle": 1, "Hesitant": 2, "Neutral": 3}
        self.null_token = 4
        
        # Load styles from JSONL
        self.episode_styles = {}
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data["final_style"] != "Discarded":
                        self.episode_styles[data["episode_index"]] = data["final_style"]
                    
        self.valid_indices = []
        self.index_to_style = []
        self.weights = []
        
        counts = Counter()
        
        # Build valid indices
        for ep in self.ds.meta.episodes:
            ep_idx = ep['episode_index']
            
            if ep_idx not in self.episode_styles:
                continue
                
            st = self.episode_styles[ep_idx]
            
            start_idx = ep['dataset_from_index']
            end_idx = ep['dataset_to_index']
            
            for i in range(start_idx, end_idx - self.seq_len + 1):
                self.valid_indices.append(i)
                self.index_to_style.append(self.style_map[st])
                counts[st] += 1
                
        # Calculate inverse class weights for rebalancing
        print(f"Loaded Style Distribution: {counts}")
        weight_per_class = {self.style_map[k]: 1.0 / v for k, v in counts.items()}
        for st_idx in self.index_to_style:
            self.weights.append(weight_per_class[st_idx])
            
        # Action Normalization Bounds
        print("Computing global action min/max for [-1, 1] normalization...")
        # The underlying HuggingFace dataset stores a list of PyTorch tensors natively.
        actions = list(self.ds.hf_dataset['action'])
        actions_tensor = torch.stack(actions)
        
        self.action_min = actions_tensor.min(dim=0)[0]
        self.action_max = actions_tensor.max(dim=0)[0]
        
        print(f"Action Min: {self.action_min.tolist()}")
        print(f"Action Max: {self.action_max.tolist()}")
        
        # State Normalization Bounds
        print("Computing global state min/max for [-1, 1] normalization...")
        states = list(self.ds.hf_dataset['observation.state'])
        states_tensor = torch.stack(states)
        self.state_min = states_tensor.min(dim=0)[0]
        self.state_max = states_tensor.max(dim=0)[0]
        print(f"State Min: {self.state_min.tolist()}")
        print(f"State Max: {self.state_max.tolist()}")
            
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        obs_tensor = self.ds[real_idx]['observation.image']
        
        # Extract and normalize state
        state_tensor = self.ds[real_idx]['observation.state']
        state_tensor = 2.0 * (state_tensor - self.state_min) / (self.state_max - self.state_min) - 1.0
        
        actions = []
        for i in range(self.seq_len):
            actions.append(self.ds[real_idx + i]['action'])
        action_seq = torch.stack(actions)
        
        # Normalize to [-1, 1]
        action_seq = 2.0 * (action_seq - self.action_min) / (self.action_max - self.action_min) - 1.0
        
        style_idx = torch.tensor(self.index_to_style[idx], dtype=torch.long)
        return obs_tensor, state_tensor, action_seq, style_idx

# ----------------- #
# 2. Network Architecture
# ----------------- #
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FiLM1d(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, channels * 2)
        )
    def forward(self, x, cond):
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(2) 
        beta = beta.unsqueeze(2)   
        return x * (1 + gamma) + beta

class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.film = FiLM1d(out_channels, cond_dim)
        
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        
        self.act = nn.Mish()
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, cond)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.residual_conv(x)

class DiffPolicy(nn.Module):
    def __init__(self, obs_dim=512, action_dim=2, style_dim=128, time_dim=128):
        super().__init__()
        
        # Visual Encoder (ResNet-18)
        resnet = models.resnet18(weights='DEFAULT')
        self.visual_encoder = nn.Sequential(
            *list(resnet.children())[:-1], 
            nn.Flatten(),
            nn.LayerNorm(512)
        )
        
        # 1D Positional Embedding to break translation invariance of CNNs
        self.action_pos_emb = nn.Parameter(torch.randn(1, action_dim, 16))
        
        # State Encoder
        self.state_emb = nn.Sequential(
            nn.Linear(2, 64),
            nn.Mish(),
            nn.Linear(64, 64)
        )
        
        # Style Encoder
        self.style_emb = nn.Embedding(5, style_dim) # 4 classes + 1 NULL
        
        # Time Encoder
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        cond_dim = obs_dim + 64 + style_dim + time_dim
        
        # 1D CNN U-Net Backbone (Wider for rapid convergence)
        self.down1 = ResBlock1d(action_dim, 128, cond_dim, kernel_size=5)
        self.down2 = ResBlock1d(128, 256, cond_dim, kernel_size=5)
        self.mid = ResBlock1d(256, 256, cond_dim, kernel_size=15)
        self.up1 = ResBlock1d(256 + 256, 128, cond_dim, kernel_size=5)
        self.up2 = ResBlock1d(128 + 128, 128, cond_dim, kernel_size=5)
        
        self.final_conv = nn.Conv1d(128, action_dim, 1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)
        
    def forward(self, noisy_action, obs, state, style_idx, t):
        # 1. Visual Features
        o_cond = self.visual_encoder(obs) 
        
        # State Features
        st_cond = self.state_emb(state)
        
        # 2. Style Features
        s_cond = self.style_emb(style_idx) 
        
        # 3. Time Features
        t_cond = self.time_emb(t) 
        
        # Concatenate conditions and inject
        cond = torch.cat([o_cond, st_cond, s_cond, t_cond], dim=-1) 
        
        # Action sequence
        x = noisy_action.transpose(1, 2)
        x = x + self.action_pos_emb 
        
        # U-Net
        d1 = self.down1(x, cond)
        d2 = self.down2(d1, cond)
        m = self.mid(d2, cond)
        
        u1 = self.up1(torch.cat([m, d2], dim=1), cond)
        u2 = self.up2(torch.cat([u1, d1], dim=1), cond)
        
        out = self.final_conv(u2)
        return out.transpose(1, 2)

# ----------------- #
# 3. Diffusion Process
# ----------------- #
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DDPM(nn.Module):
    def __init__(self, model, timesteps=100):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
    def forward(self, clean_action, obs, state, style_idx, p_drop=0.1):
        B = clean_action.shape[0]
        device = clean_action.device
        
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        noise = torch.randn_like(clean_action)
            
        sqrt_alpha_m = self.alphas_cumprod[t].sqrt().view(B, 1, 1)
        sqrt_one_minus_alpha_m = (1. - self.alphas_cumprod[t]).sqrt().view(B, 1, 1)
        noisy_action = sqrt_alpha_m * clean_action + sqrt_one_minus_alpha_m * noise
        
        # Random condition dropout for CFG
        if self.training and p_drop > 0.0:
            dropout_mask = torch.rand(B, device=device) < p_drop
            style_idx = style_idx.clone()
            style_idx[dropout_mask] = 4 
            
        noise_pred = self.model(noisy_action, obs, state, style_idx, t)
        loss = F.mse_loss(noise_pred, noise)
        return loss, noise_pred, noise

# ----------------- #
# 4. Utilities (EMA)
# ----------------- #
class EMAModel:
    """Exponential Moving Average of model weights"""
    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        self.model.requires_grad_(False)
        
    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=(1 - self.decay))

# ----------------- #
# 5. Full Training Loop
# ----------------- #
def train():
    device = torch.device("mps")
    print(f"Starting Training on Device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 300
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    P_DROP = 0.1
    CHECKPOINT_DIR = "checkpoints"
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "policy_latest.pth")
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "policy_best.pth")
    
    # Dataset & DataLoader
    dataset = StylizedPushTDataset(seq_len=16)
    sampler = WeightedRandomSampler(dataset.weights, num_samples=len(dataset.weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True, num_workers=0)
    
    # Models
    policy = DiffPolicy().to(device)
    ddpm = DDPM(policy).to(device)
    ema = EMAModel(policy)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # State tracking
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint if it exists
    if os.path.exists(latest_ckpt_path):
        print(f"--> Found checkpoint at {latest_ckpt_path}. Resuming training...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        ema.model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"--> Resumed from Epoch {start_epoch} with Best Loss {best_loss:.4f}")

    print("--- Starting Training Loop ---")
    for epoch in range(start_epoch, EPOCHS):
        policy.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for obs_batch, state_batch, action_batch, style_batch in progress_bar:
            obs_batch = obs_batch.to(device)
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            style_batch = style_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with p_drop=0.1 for CFG
            loss, _, _ = ddpm(action_batch, obs_batch, state_batch, style_batch, p_drop=P_DROP)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            # Update EMA weights
            ema.update(policy)
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1:04d}/{EPOCHS} | Avg Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")
        
        # Save exact latest state every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'ema_state_dict': ema.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, latest_ckpt_path)
            
        # Track and save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Best model primarily values the EMA weights for clean inference
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema.model.state_dict(), 
                'best_loss': best_loss,
            }, best_ckpt_path)
            print(f" >> Saved new Best Model (Loss: {best_loss:.5f}) to {best_ckpt_path}")

if __name__ == "__main__":
    train()
