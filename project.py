import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadWebcam, LoadImages
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
from PIL import Image
import onnxruntime as ort
from utils.emotion import HSEmotionRecognizer


providers = (
    ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
)
fer=HSEmotionRecognizer('emotion-model.onnx',providers=providers)
age_session = ort.InferenceSession("classification_model.onnx", providers=providers)
gender_session = ort.InferenceSession("age_gender_model.onnx", providers=providers)


GENDER_DICT = {0: "Male", 1: "Female"}



def predict_attributes(image):
    # cv2.imshow("image", image)
    # Görüntüyü uygun boyutlara göre yeniden boyutlandırın
    resized_img = cv2.resize(image, (64, 64))

    # Veriyi işleyin: Kanal sırasını (H, W, C) -> (C, H, W) dönüştürün ve modele uygun hale getirin
    input_data = np.transpose(resized_img, (2, 0, 1))  # (64, 64, 3) -> (3, 64, 64)
    input_data = np.expand_dims(input_data, 0).astype(np.float32) / 255.0  # (1, 3, 64, 64)

    # Cinsiyet ve yaş tahmini için gender_session modelini çalıştırın
    gender_predictions = gender_session.run(None, {gender_session.get_inputs()[0].name: input_data})[0]
    gender_pred = int(np.argmax(gender_predictions[0, :2]))  # İlk iki eleman cinsiyet için
    age_pred = int(gender_predictions[0, 2])  # Yaşı tam sayıya çevirerek alıyoruz

    # Duygu tahmini için emotion_session modelini çalıştırın
    emotion,scores=fer.predict_emotions(image,logits=True)

    return GENDER_DICT[gender_pred], age_pred, emotion


# def predict_emotions(self, face_img, logits=True):
#         scores = self.ort_session.run(None, {"input": self.preprocess(face_img)})[0][0]
#         x = scores
#         pred = np.argmax(x)
#         if not logits:
#             e_x = np.exp(x - np.max(x)[np.newaxis])
#             e_x = e_x / e_x.sum()[None]
#             scores = e_x
#         return self.idx_to_class[pred], scores


################################
# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid


def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


realWidth = 320
realHeight = 240
videoWidth = int(realWidth * 0.75)
videoHeight = int(realHeight / 2)
w_pad = int((realWidth - videoWidth) / 2)
h_pad = int((realHeight - videoHeight) / 2)

videoChannels = 3
videoFrameRate = 10

# Color Magnification Parameters

levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Output Display Parameters

font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth // 2 + 5, 30)
fontScale = 1
fontColor = (0, 255, 0)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid

firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros(
    (bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels)
)
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies

frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables

bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

ind = 0
################################
palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
data_deque = {}
wh_deque = {}


def xyxy_to_xywh(*xyxy):
    """ " Calculates the relative bounding box from absolute pixel values."""
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = bbox_left + bbox_w / 2
    y_c = bbox_top + bbox_h / 2
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0:  # person  #BGR
        color = (131, 59, 236)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    elif label == 7:  # truck
        color = (85, 45, 255)
    else:
        color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(
            img,
            (c1[0], c1[1] - t_size[1] - 3),
            (c1[0] + t_size[0], c1[1] + 3),
            color,
            1,
            8,
            2,
        )

        cv2.rectangle(img, c1, c2, color, tl, cv2.LINE_AA)  # not filled

        xr = c2[0] - c1[0]
        if xr > 0:

            xr = int(0.1 * xr)
        else:
            xr = 20

        yr = c2[1] - c1[1]
        if yr > 0:

            yr = int(0.1 * yr)
        else:
            yr = 20
        cv2.line(
            img, pt1=c1, pt2=(c1[0] + xr, c1[1]), color=(219, 134, 59), thickness=5
        )
        cv2.line(
            img, pt1=c1, pt2=(c1[0], c1[1] + yr), color=(219, 134, 59), thickness=5
        )

        cv2.line(
            img, pt1=(c2[0] - xr, c2[1]), pt2=c2, color=(219, 134, 59), thickness=5
        )
        cv2.line(
            img,
            pt1=(c2[0], c1[1]),
            pt2=(c2[0], c1[1] + yr),
            color=(219, 134, 59),
            thickness=5,
        )

        cv2.line(
            img,
            pt1=(c1[0], c2[1]),
            pt2=(c1[0] + xr, c2[1]),
            color=(219, 134, 59),
            thickness=5,
        )
        cv2.line(
            img,
            pt1=(c1[0], c2[1] - yr),
            pt2=(c1[0], c2[1]),
            color=(219, 134, 59),
            thickness=5,
        )

        cv2.line(
            img,
            pt1=(c2[0] - xr, c1[1]),
            pt2=(c2[0], c1[1]),
            color=(219, 134, 59),
            thickness=5,
        )
        cv2.line(
            img,
            pt1=(c2[0], c2[1] - yr),
            pt2=(c2[0], c2[1]),
            color=(219, 134, 59),
            thickness=5,
        )

        ##text Class

        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def draw_boxes(img, bbox, object_id, names, identities=None, offset=(0, 0)):
    height, width, _ = img.shape
    calibration_constant = 0.01
    # Remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)
    for key in list(wh_deque):
        if key not in identities:
            wh_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        top = y1
        left = x1
        box_height = y2 - y1

        # Center of the bottom edge
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
        wh = ((x2 - x1), (y2 - y1))

        # Get ID of object
        id = int(identities[i]) if identities is not None else 0

        # Create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=opt.trailslen)
        if id not in wh_deque:
            wh_deque[id] = deque(maxlen=64)

        color = compute_color_for_labels(object_id[i])
        label = "%s" % (names[object_id[i]])

        # Add center to buffer
        data_deque[id].appendleft(center)
        # Add wh to buffer
        wh_deque[id].appendleft(wh)

        # Crop Image
        xc = int((x2 + x1) / 2)
        yc = int((y2 + y1) / 2)
        w_mean = sum(w for w, h in wh_deque[id]) / len(wh_deque[id])
        h_mean = sum(h for w, h in wh_deque[id]) / len(wh_deque[id])

        xmin = max(0, int(xc - w_mean / 2))
        xmax = min(width, int(xc + w_mean / 2))
        ymin = max(0, int(yc - h_mean / 2))
        ymax = min(height, int(yc + h_mean / 2))

        # crop_img_orig = img[ymin:ymax, xmin:xmax]
        crop_img =img[ymin:ymax, xmin:xmax]
        dim = (realWidth, realHeight)
        crop_img = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
        crop_img_orig = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)


        # Heart Rate Detection
        global bufferIndex, ind, bpmBufferIndex

        frame = crop_img.copy()
        detectionFrame = frame[h_pad : realHeight - h_pad, w_pad : realWidth - w_pad, :]

        # Construct Gaussian Pyramid
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Bandpass Filter
        fourierTransform[mask == False] = 0

        # Grab a Pulse
        if bufferIndex % bpmCalculationFrequency == 0:
            ind = ind + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Amplify
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        # Reconstruct Resulting Frame
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)
        cv2.imshow("Filtered Frame", outputFrame)
        bufferIndex = (bufferIndex + 1) % bufferSize

        # frame[h_pad : realHeight - h_pad, w_pad : realWidth - w_pad, :] = outputFrame
        cv2.rectangle(
            frame,
            (w_pad, h_pad),
            (realWidth - w_pad, realHeight - h_pad),
            boxColor,
            boxWeight,
        )

        # Display BPM if available
        if ind > bpmBufferSize:
            cv2.putText(
                frame,
                "BPM: %d" % bpmBuffer.mean(),
                bpmTextLocation,
                font,
                fontScale,
                fontColor,
                lineType,
            )
        else:
            cv2.putText(
                frame,
                "Calculating BPM...",
                loadingTextLocation,
                font,
                fontScale,
                fontColor,
                lineType,
            )

        crop_img = frame.copy()

        # Prediction for Gender, Age, and Emotion
        gender_pred, age_pred, emotion_pred = predict_attributes(crop_img_orig)
        # Add attributes text to the frame
       
        
        cv2.putText(
            img,
            f'Gender: {gender_pred}',
            (left, top - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            f'Age: {str(age_pred)}',
            (left, top - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            f'Emotion: {emotion_pred}',
            (left, top - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
       

        x_offset = width - 320
        y_offset = 5 + 230 * list(wh_deque.keys()).index(id)
        y_end = y_offset + crop_img.shape[0]
        x_end = x_offset + crop_img.shape[1]

        if y_end > img.shape[0]:
            y_end = img.shape[0]
        if x_end > img.shape[1]:
            x_end = img.shape[1]

        # img[y_offset:y_end, x_offset:x_end] = crop_img[
        #     : y_end - y_offset, : x_end - x_offset
        # ]
        cv2.imshow("Cropped", crop_img)
        # Draw Bounding Box
        UI_box(box, img, label=label, color=color, line_thickness=2)

        # Draw trail
        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue

            thickness = int(np.sqrt(opt.trailslen / float(i + i)) * 1.5)

        # Display ID
        label = "{}{:d}".format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1
        )
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            [255, 255, 255],
            1,
        )

    return img


def detect(save_img=False):

    source, weights, view_img, save_txt, imgsz, trace = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
        not opt.no_trace,
    )
    save_img = not opt.nosave and not source.endswith(".txt")  # save inference images
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # initialize deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        n_init=cfg_deep.DEEPSORT.N_INIT,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=True,
    )

    # Load model
    model = attempt_load(weights, map_location=device)  
    stride = int(model.stride.max()) 
    imgsz = check_img_size(imgsz, s=stride)  

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadWebcam(int(source), img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    t0 = time.time()
    prevTime = 0
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
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], "%g: " % i, im0s.copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)

            if not webcam:
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / "labels" / p.stem) + (
                    "" if dataset.mode == "image" else f"_{frame}"
                )  # img.txt
            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywh_bboxs = []
                confs = []
                oids = []
                # Write results
                for *xyxy, conf, cls in reversed(det):

                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    label = "%s" % (names[int(cls)])
                    # color = compute_color_for_labels(int(cls))
                    # UI_box(xyxy, im0, label=label, color=color, line_thickness=2)
                    oids.append(int(cls))

                    if not webcam:
                        if save_txt:  # Write to file
                            xywh = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                                .view(-1)
                                .tolist()
                            )  # normalized xywh
                            line = (
                                (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            )  # label format
                            with open(txt_path + ".txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                    xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).tolist()
                    xc, yc = int(xywh[0][0]), int(xywh[0][1])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                outputs = deepsort.update(xywhs, confss, oids, im0)

                if len(outputs) > 0:

                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]

                    draw_boxes(im0, bbox_xyxy, object_id, names, identities)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.line(im0, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(
                im0,
                f"FPS: {int(fps)}",
                (11, 35),
                0,
                1,
                [225, 255, 255],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)

                # 1 millisecond

            # Save results (image with detections)
            if not webcam:
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
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
                                save_path += ".mp4"
                            vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                            )
                        vid_writer.write(im0)

        stop = cv2.waitKey(1) == ord("q")

        if webcam and stop:
            break
    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="website_elements.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--source", type=str, default="inference/images", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")
    parser.add_argument(
        "--trailslen", type=int, default=64, help="trails size (new parameter)"
    )
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov7.pt"]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
