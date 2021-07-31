import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os
from PIL import Image
import numpy as np


@torch.no_grad()
def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
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
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                im0 = Image.fromarray(np.uint8(im0))  # 使用pillow打开图片，是图片的颜色保持正常
                # im0.show()
                # cv2.imshow(str(p), im0) # cv2打开的图片发蓝
                # cv2.waitKey(100000)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    im0.save(save_path)  # 图片会被保存再runs_chengchong\exp4\detect路径下，图片名即原本的图片命
                    # cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':

    dir_num = None  # 选择使用哪一个文件夹里的模型，根据runs文件夹下 saved_model文件夹的编号自己选择

    source = './img'  # 选择存放测试图片的文件夹，source参数可以是一个装有图片的文件夹（预测整个文件夹内的图片），也可以是单张图片
    # source = './img/000_Pw3qUBVmwN8.jpg'
    # weights = f'./runs/saved_model{dir_num}/weights/best.pt'  # 选择该文件夹下的模型
    weights = f'./runs/saved_model/weights/best.pt'  # 选择该文件夹下的模型
    project = './predication/'.format(type)  # 预测图片存放的位置


    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model.pt path(s)')  #模型权重位置
    parser.add_argument('--source', type=str, default=source, help='source')  # file/folder  可以是图片/视频路径，也可以是'0'(电脑自带摄像头)，也可以是rtsp等视频流
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)') #网络输入图片大小
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold') #置信度阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS') #做nms的iou阈值
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image') #每张图片上允许的最多的预测框的数量
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') #使用什么设备预测
    parser.add_argument('--view-img', default=True,action='store_true', help='display results') #是否展示预测之后的图片/视频，默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt') #是否将预测的框坐标以txt文件形式保存，这里设置的是文件名，默认False
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #是否将预测框坐标以txt文件形式保存，默认False
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes') #
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos') #
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3') #设置只保留某一部分类别，形如0或0，2，3
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS') # 进行nms是否也去除不同类别之间的框，默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference') #推理的时候进行多尺度，翻转等操作(TTA)推理，指与测试进行数据增强
    parser.add_argument('--update', action='store_true', help='update all models')  #如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器信息，默认为False
    parser.add_argument('--project', default=project, help='save results to project/name')  # 图片检测完成后存放的目录
    parser.add_argument('--name', default='detect', help='save results to project/name')  # 图片预测完成后存放的文件夹
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=0, type=int, help='bounding box thickness (pixels)')  # 调整预测框线宽，设为0 则自动根据图片尺寸调整
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # 是否隐藏预测标签
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') # 是否隐藏预测的置信度
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
