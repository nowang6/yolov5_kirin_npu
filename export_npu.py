import argparse
import sys
from pathlib import Path
import time
import torch
import os

# Add YOLOv5 root directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.yolo import DetectionModel
from utils.general import LOGGER, check_img_size, check_requirements, print_args
from utils.torch_utils import select_device, smart_inference_mode


def export_torch(model, file, prefix='export'):
    # YOLOv5 PyTorch export
    try:
        LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = str(file).replace('.pt', '_torch.pt')  # convert weights path to torch path

        # Save complete model with configuration
        save_dict = {
            'model': model,  # complete model
            'model_yaml': model.yaml,  # model configuration
            'names': model.names,  # class names
            'stride': model.stride,  # model stride
        }
        torch.save(save_dict, f)
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix='export'):
    # YOLOv5 ONNX export
    model.eval()
    try:
        check_requirements(('onnx', 'onnxruntime'))
        import onnx

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = str(file).replace('.pt', '.onnx')  # convert weights path to onnx path

        torch.onnx.export(
            model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic else im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'},  # shape(1,3,640,640)
                'output': {
                    0: 'batch',
                    1: 'anchors'}  # shape(1,25200,85)
            } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        
        # Remove additional outputs
        if len(model_onnx.graph.output) > 1:
            LOGGER.info(f'{prefix} removing additional Sigmoid outputs...')
            while len(model_onnx.graph.output) > 1:
                model_onnx.graph.output.pop()
        
        onnx.checker.check_model(model_onnx)  # check onnx model
        onnx.save(model_onnx, f)  # save modified model

        # Simplify
        if simplify:
            try:
                import onnxsim
                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MB (1024 * 1024)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('onnx', 'torch'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        optimize=False,  # TorchScript: optimize for mobile
        simplify=False,  # ONNX: simplify model
        opset=10,  # ONNX: opset version
        dynamic=False,  # ONNX/TensorRT: dynamic axes
        verbose=False,  # TensorRT: verbose log
        workspace=4,  # TensorRT: workspace size (GB)
        nms=False,  # TF: add NMS to model
        agnostic_nms=False,  # TF: agnostic NMS
        topk_per_class=100,  # TF.js NMS: topk per class to keep
        topk_all=100,  # TF.js NMS: topk for all classes to keep
        iou_thres=0.45,  # TF.js NMS: IoU threshold
        conf_thres=0.25,  # TF.js NMS: confidence threshold
):
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(include) if len(include) > 1 else include[0]  # export formats
    flags = [inplace, optimize]  # export flags
    device = select_device(device)

    # Load PyTorch model
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    model = DetectionModel(cfg='models/yolov5s.yaml', ch=3, nc=80).to(device)  # create model with NPU config
    model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)  # load state_dict
    stride = int(max(model.stride))  # convert stride to int
    names = model.names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Input
    gs = stride  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
    for f in flags:
        model = model.fuse() if f else model  # model inplace

    # Export
    if 'onnx' in include:
        export_onnx(model, im, weights, opset, train, dynamic, simplify)
    if 'torch' in include:
        export_torch(model, weights)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (height, width)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include', nargs='+', default=['onnx', 'torch'], help='include formats')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--simplify', action='store_true', default=True, help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=10, help='ONNX: opset version')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TensorRT: dynamic axes')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    parser.add_argument('--nms', action='store_true', help='TF: add NMS to model')
    parser.add_argument('--agnostic-nms', action='store_true', help='TF: agnostic NMS')
    parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
    parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
