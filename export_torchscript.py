import argparse

import torch

from src.models import CenterNetDetectionTorchScript

def export(checkpoint_path, save_path, input_size=512):
    print("Loading model")
    model = CenterNetDetectionTorchScript.load_centernet_from_checkpoint(checkpoint_path)
    model.eval()

    print("Converting TorchScript")
    script = torch.jit.trace(model, torch.rand(1,3,input_size,input_size))

    print("Saving TorchScript file")
    torch.jit.save(script, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CenterNet Object Detector to TorchScript format using tracing")
    parser.add_argument("--checkpoint", type=str, help="path to Lightning checkpoint")
    parser.add_argument("--save-path", type=str, help="path to save TorchScript file")
    parser.add_argument("--input-size", default=512, type=int, help="input image size for tracing")
    args = parser.parse_args()

    export(args.checkpoint, args.save_path, args.input_size)
