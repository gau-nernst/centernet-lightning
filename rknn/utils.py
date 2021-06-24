import os

import cv2
from rknn.api import RKNN

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def resize_images(img_dir, output_dir, img_names=None, resize_dim=512):
    if img_names is None:
        img_names = os.listdir(img_dir)
    
    for name in img_names:
        img_path = os.path.join(img_dir, name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (resize_dim, resize_dim))
        
        save_path = os.path.join(output_dir, name)
        success = cv2.imwrite(save_path, img)
        assert success

def torchscript_to_rknn(torchscript_path, rknn_path, input_size=512, do_quantization=False, quantization_dataset=None):
    rknn = RKNN()

    imagenet_mean_uint8 = [x*255 for x in IMAGENET_MEAN]
    imagenet_std_uint8 = [sum(IMAGENET_STD)/3*255] * 3
    rknn.config(
        batch_size=4,
        mean_values=[imagenet_mean_uint8],
        std_values=[imagenet_std_uint8],
        epochs=-1,
        reorder_channel='0 1 2',
        quantized_dtype="asymmetric_quantized-u8",
        quantized_algorithm="quantized_algorithm",
        # mmse_epoch=3,
        optimization_level=3,
        target_platform=["rk1808"]
    )

    print("Loading TorchScript model")
    rknn.load_pytorch(model=torchscript_path, input_size_list=[[3, input_size, input_size]])

    print("Building RKNN model")
    rknn.build(
        do_quantization=do_quantization,
        dataset=quantization_dataset,
        rknn_batch_size=1,
    )

    print("Saving RKNN model")
    rknn.export_rknn(rknn_path)
    rknn.release()

def rknn_inference(rknn_path, img_path, resize_img=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize_img is not None:
        img = cv2.resize(img, (resize_img, resize_img))
    
    rknn = RKNN()
    rknn.load_rknn(rknn_path)
    rknn.init_runtime()

    outputs = rknn.inference(inputs=[img])[0]
    rknn.release()

    return outputs

def rknn_fps_benchmark(rknn_path, img_path, resize_img=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize_img is not None:
        img = cv2.resize(img, (resize_img, resize_img))
    
    rknn = RKNN()
    rknn.load_rknn(rknn_path)
    rknn.init_runtime(perf_debug=True)

    results = rknn.eval_perf(inputs=[img])
    rknn.release()

    return results

def rknn_memory_benchmark(rknn_path):
    rknn = RKNN()
    rknn.load_rknn(rknn_path)
    rknn.init_runtime(eval_mem=True)

    results = rknn.eval_memory()
    rknn.release()

    return results
