# YOLOv5 for Kirin NPU

This repository contains a modified version of [YOLOv5](https://github.com/ultralytics/yolov5) optimized for inference on Kirin NPU. The main modifications focus on adapting the detection head structure to avoid using 5-dimensional tensors, making it compatible with Kirin NPU hardware.

## Key Modifications

- Modified the detection head in `models/yolo.py` to be compatible with Kirin NPU
- Restructured tensor operations to avoid 5D tensors
- Added NPU-specific inference and export scripts

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch >= 1.7.0
- Required packages from original YOLOv5 repository

### Installation

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/yolov5_kirin_npu
cd yolov5_kirin_npu
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained YOLOv5s weights:
```bash
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

## Usage

### Running Inference with NPU

To test the model inference on Kirin NPU:

```bash
python detect_npu.py
```

### Exporting to ONNX

To export the model to ONNX format for NPU deployment:

```bash
python export_npu.py
```

### Testing ONNX Model

To verify the exported ONNX model:

```bash
python detect_onnx_npu.py
```

## Model Architecture Changes

The main modifications are in the detection head implementation (`models/yolo.py`). The changes include:
- Restructured tensor operations to maintain compatibility with Kirin NPU
- Optimized detection head to avoid 5D tensor operations
- Modified output format while maintaining detection accuracy

## Performance

[Add any performance metrics, comparison with original model, or NPU-specific benchmarks]

## Acknowledgments

- Original YOLOv5 implementation by [Ultralytics](https://github.com/ultralytics/yolov5)
- [Add any other acknowledgments]

## License

This project inherits the AGPL-3.0 license from the original YOLOv5 repository. See the [LICENSE](LICENSE) file for details.

