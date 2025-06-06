import torch
import cv2
import numpy as np
from models.yolo_npu import DetectionModelNPU
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

from utils.torch_utils import  smart_inference_mode

@smart_inference_mode()
def main():
    # 1. 加载模型
    model = DetectionModelNPU(cfg='models/yolov5s_npu.yaml', ch=3, nc=80)  # 使用YOLOv5s配置
    
    # 加载预训练权重
    weights = 'yolov5s.pt'  # 预训练权重文件
    ckpt = torch.load(weights, map_location='cpu')  # 加载权重
    model.load_state_dict(ckpt['model'].state_dict())  # 将权重加载到模型
    model.eval()  # 设置为评估模式
    
    # 2. 加载图片
    img_path = 'data/images/zidane.jpg'  # 使用示例图片
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. 图像预处理
    img_size = (640, 640)  # 固定输入尺寸
    img_resized = cv2.resize(img, img_size)
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float()  # HWC to CHW
    img_tensor /= 255.0  # 归一化到0-1
    img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
    
    # 4. 推理
    with torch.no_grad():
        pred = model(img_tensor)
    
    # 5. 后处理
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    
    # 6. 可视化结果
    annotator = Annotator(img, line_width=3, example=str(model.names))
    
    # 处理检测结果
    for i, det in enumerate(pred):
        if len(det):
            # 将边界框从img_size缩放到原始图像大小
            det[:, :4] = scale_boxes(img_size, det[:, :4], img.shape).round()
            
            # 绘制检测框和标签
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{model.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
    
    # 7. 显示结果
    result_img = annotator.result()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Detection Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
