import argparse

import torch

from centernet_lightning.models import CenterNet

def export_torchscript(checkpoint_path, save_path, input_size=512):
    model = CenterNet.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to_torchscript(save_path, method="trace", example_inputs=torch.rand(1,3,input_size,input_size))
    
    print("Done")

def export_onnx(checkpoint_path, save_path, input_size=512):
    model = CenterNet.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to_onnx(save_path, example_inputs=torch.rand(1,3,input_size,input_size), opset_version=11)
    
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CenterNet to TorchScript or ONNX")
    parser.add_argument("format", type=str, help="torchscript or onnx")
    parser.add_argument("--checkpoint", type=str, help="path to Lightning checkpoint")
    parser.add_argument("--save-path", type=str, help="path to save TorchScript/ONNX file")
    parser.add_argument("--input-size", type=int, default=512, help="input image size for tracing")
    args = parser.parse_args()

    if args.format == "torchscript":
        export_torchscript(args.checkpoint, args.save_path, args.input_size)
    elif args.format == "onnx":
        export_onnx(args.checkpoint, args.save_path, args.input_size)
