import os
import json
from collections import Counter

import torch
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError

# Setup Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

def tensor_to_pil(img_tensor):
    """
    Convert image Tensor to a PIL Image.
    Handles [C, H, W] to [H, W, C] seamlessly.
    """
    return F.to_pil_image(img_tensor)

# 1. Robust API Retries with logic for rate limits and server errors
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, InternalServerError))
)
def analyze_window(window_images, prompt):
    """
    Passes the sliding window sequence to the Gemini API and parses the JSON response.
    Retries automatically with exponential backoff if 429 or 503 errors occur.
    """
    response = model.generate_content(
        window_images + [prompt],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json"
        )
    )
    return json.loads(response.text)

def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print("Loading LeRobotDataset 'lerobot/pusht'...")
    dataset = LeRobotDataset("lerobot/pusht")
    
    output_file = "trajectory_styles.jsonl"
    processed_episodes = set()
    
    # 3. Incremental Checkpointing Check
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    processed_episodes.add(data["episode_index"])
        print(f"Found {len(processed_episodes)} already processed episodes. Resuming...")
        
    episodes = dataset.meta.episodes
    
    prompt = """Step 1: Analyze the robot arm's movement in terms of velocity, acceleration (jerk), and hesitation.
Step 2: Based on the analysis, classify the motion style into exactly one of: [Aggressive, Gentle, Hesitant, Neutral].
Step 3: Provide a confidence score (0.0-1.0).
Return JSON format strictly: {'analysis': '...', 'style': 'LABEL', 'confidence': 0.8}"""

    # 2. Sliding Window config
    window_size = 16
    stride = 8
    
    # 4. Progress Tracking over episodes
    for ep in tqdm(episodes, desc="Processing Trajectories"):
        ep_idx = ep['episode_index']
        
        # Skip previously processed trajectories
        if ep_idx in processed_episodes:
            continue
            
        start_idx = ep['dataset_from_index']
        end_idx = ep['dataset_to_index']
        
        trajectory_windows = []
        window_images = []
        
        # Extract images for the entire episode
        for i in range(start_idx, end_idx):
            item = dataset[i]
            img = tensor_to_pil(item['observation.image'])
            window_images.append(img)
            
        # Sliding Window Stride Adjustment (50% overlap usually)
        for i in range(0, len(window_images) - window_size + 1, stride):
            trajectory_windows.append(window_images[i:i + window_size])
        
        # Fallback if trajectory is smaller than window size
        if not trajectory_windows and len(window_images) > 0:
            trajectory_windows.append(window_images)
            
        if not trajectory_windows:
            continue
            
        window_responses = []
        # Process each window via API
        for w_imgs in trajectory_windows:
            try:
                resp_json = analyze_window(w_imgs, prompt)
                window_responses.append(resp_json)
            except Exception as e:
                # Capture unrecoverable errors after retries and log them nicely
                tqdm.write(f"Error processing a window in episode {ep_idx}: {e}")
                
        if not window_responses:
            continue
            
        styles = [resp.get('style', 'Neutral') for resp in window_responses]
        confidences = [float(resp.get('confidence', 0.0)) for resp in window_responses]
        
        # Majority voting
        style_counts = Counter(styles)
        majority_style = style_counts.most_common(1)[0][0]
        
        # Keep confidence of only the predictions that match the chosen majority
        mode_confidences = [confidences[i] for i, s in enumerate(styles) if s == majority_style]
        mean_mode_conf = sum(mode_confidences) / len(mode_confidences) if mode_confidences else 0.0
            
        final_label = majority_style
        if mean_mode_conf < 0.6:
            final_label = "Discarded"
            
        # 3. Incremental Checkpointing (write instantly)
        result = {
            "episode_index": ep_idx,
            "final_style": final_label,
            "confidence": mean_mode_conf
        }
        
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main()
