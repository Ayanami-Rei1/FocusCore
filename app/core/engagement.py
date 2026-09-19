"""
Engagement model inference

Loads the TorchScript model and runs video frames through it to predict
engagement 
"""

from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
import sys

MODEL_PATH = Path(__file__).parent.parent / "models" / "engagement_slow_r50.ts"
NUM_FRAMES = 8
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class EngagementModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model = torch.jit.load(str(model_path))
        self.model.eval()

    def _preprocess_clip(self, frames_bgr):
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        clip = np.ascontiguousarray(np.stack(rgb))

        t = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0
        t = TF.resize(t, 256)
        t = TF.center_crop(t, [IMAGE_SIZE, IMAGE_SIZE])
        t = TF.normalize(t, MEAN, STD)
        t = t.permute(1, 0, 2, 3)
        t = t.unsqueeze(0)
        return t
    
    def predict(self, frames_bgr):
        clip = self._preprocess_clip(frames_bgr)
        with torch.no_grad():
            logits = self.model(clip)
            probs = torch.softmax(logits, dim=1)[0]
        return probs

    def engagement_score(self, frames_bgr):
        probs = self.predict(frames_bgr)
        weights = torch.tensor([0.0, 50.0, 100.0])
        return float((probs * weights).sum())

if __name__ == "__main__":
    model = EngagementModel()

    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    if video_path is None:
        fake = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                for _ in range(NUM_FRAMES)]
        
        print("probs:", model.predict(fake))
    else:
        STRIDE = 8
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        window = NUM_FRAMES * STRIDE
        start = max(0, (total - window) // 2)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        pos = 0
        needed = {i * STRIDE for i in range(NUM_FRAMES)}
        last = max(needed)
        while pos <= last:
            grabbed = cap.grab()
            if pos in needed:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append(frame)
            pos += 1
        cap.release()

        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1])

        probs = model.predict(frames)

        print("probs [low, mid, high]:", probs.tolist())
        print("pred class:", int(probs.argmax()))