import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from detection.models.experimental import attempt_load
from detection.utils.datasets import LoadStreams, LoadImages
from detection.utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from detection.utils.plots import colors, plot_one_box
from detection.utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(opt):
    source, weights, imgsz = opt.source, opt.weights, opt.img_size

    # Directories
    save_dir = Path(opt.dst)

    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # model stride
    stride = int(model.stride.max())
    # check img_size
    imgsz = check_img_size(imgsz, s=stride)
    # get class names
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()  # to FP16

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / f"det_{p.name}")  # Add Prefix det_
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{names[int(c)]}{'s' * (n > 1)}, "  # add to string

                obj = []

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = names[c]
                    obj.append(names[c])
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            cv2.imwrite(save_path, im0)
    print(f"Results saved to {save_dir}")
    print(f'Done. ({time.time() - t0:.3f}s)')
    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        print(detect(opt=opt))
