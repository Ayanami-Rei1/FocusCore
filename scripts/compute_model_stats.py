"""Print the layer summary, parameter count and MACs of a registered model.

Usage:
    python scripts/compute_model_stats.py --model slow_r50 --batch_size 8 --frames_count 8
"""

import argparse
from torchinfo import summary
import src.models.backbones.pytorchvideo
from src.models.factory import create_model

def main():
    parser = argparse.ArgumentParser(description="Model summary with torchinfo")
    parser.add_argument("--model", type=str, default="slow_r50")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--frames_count", type=int, default=8)
    args = parser.parse_args()

    model = create_model(name=args.model)
    summary(model, input_size=(args.batch_size, 3, args.frames_count, 224, 224))

if __name__ == "__main__":
    main()
