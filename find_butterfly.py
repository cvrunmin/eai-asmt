from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import cv2 as cv
from contextlib import contextmanager
import argparse
import os.path
import numpy as np
import time

def getIoU(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """ get IoU (Intersection over Union) score between two boxes

    Args:
        box1 (torch.Tensor): box 1
        box2 (torch.Tensor): box 2

    Returns:
        float: IoU score
    """
    x1, y1, x2, y2 = box1.tolist()
    x3, y3, x4, y4 = box2.tolist()
    area1 = (x2-x1)*(y2-y1)
    area2 = (x4-x3)*(y4-y3)
     
    # find intersection box
    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)
 
    # two boxes are overlapped if x5 < x6 and y5 < y6. Else we have zero width/height here
    w = max(0, x6 - x5)
    h = max(0, y6 - y5)
 
    intersection_area = w*h
 
    union_area = area1 + area2 - intersection_area
 
    return intersection_area / union_area

@contextmanager
def ManagedVideoCapture(*args, **kwargs):
    """ Helper function to make cv.VideoCapture release in more pythonic way
    """
    video = cv.VideoCapture(*args, **kwargs)
    try:
        yield video
    finally:
        video.release()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Video file')
    parser.add_argument('--score_threshold', default=0.35, help='Threshold of box scores. Any box with scores'\
        ' lower than the threshold will be discarded')
    parser.add_argument('--nms_threshold', default=0.3, help='Threshold for non max suppression. Boxes with IoU'\
        ' higher than the threshold will be discarded')
    parser.add_argument('--cuda', action='store_true', help='Use cuda gpu accelaration if possible. No-op for no usable cuda device')
    args = parser.parse_args()
    filepath = args.file
    score_threshold = args.score_threshold    
    nms_threshold = args.nms_threshold
    if not os.path.isfile(filepath):
        print(f'file {filepath} not found')
        exit(1)
        
    torch.set_grad_enabled(False)
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Loading pretrained models')
    processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    texts = [["butterfly"]]
    
    print('Start detecting')
    last_count = -1
    last_diff = 0
    last_diff_timeout = 0
    with ManagedVideoCapture(filepath) as video:
        if video.isOpened():
            fps = video.get(cv.CAP_PROP_FPS)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            img = Image.fromarray(frame)
            inputs = processor(text=texts, images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]], device=device)
            # Convert outputs (bounding boxes and class logits) to COCO API
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=score_threshold)
            boxes = list(sorted(zip(results[0]['scores'].cpu(), results[0]['boxes'].cpu()), key=lambda x: x[0], reverse=True))
            nms_boxes = []
            # Non Max Suppression
            while len(boxes) > 0:
                cur_box = boxes.pop(0)
                nms_boxes.append(cur_box)
                bad_boxes = []
                for cand_box in boxes:
                    if getIoU(cur_box[1], cand_box[1]) > nms_threshold:
                        bad_boxes.append(cand_box)
                for bad_box in bad_boxes:
                    boxes.remove(bad_box)
                del bad_boxes
            # draw boxes
            for score, xy in nms_boxes:
                cv.rectangle(frame, xy[0:2].int().numpy(), xy[2:4].int().numpy(), (255,0,0), 3)
            # extend image for texts
            frame2 = cv.copyMakeBorder(frame, 0, 50, 0, 0, cv.BORDER_CONSTANT, value = (0,0,0))
            cur_count = len(nms_boxes)
            if last_count == -1:
                diff = 0
            else:
                diff = cur_count - last_count
            butterfly_count_text = f'Butterfly: {cur_count}'
            cv.putText(frame2, butterfly_count_text, (10, frame2.shape[0] - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2, lineType=cv.LINE_AA)
            if diff != 0:
                last_diff_timeout = 5
                last_diff = diff
            elif last_diff_timeout > 0:
                last_diff_timeout -= 1
            if last_diff_timeout > 0:
                (text_width, _), _ = cv.getTextSize(butterfly_count_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
                color = int(255 * last_diff_timeout / 5)
                cv.putText(frame2, f' ({last_diff:+d})', (10 + text_width, frame2.shape[0] - 15),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (color,color,color), thickness=2, lineType=cv.LINE_AA)
            close_hint = 'Press Esc to close'
            (text_width, _), _ = cv.getTextSize(close_hint, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv.putText(frame2, close_hint, (frame2.shape[1] - 10 - text_width, frame2.shape[0] - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2, lineType=cv.LINE_AA)
            last_count = cur_count
            cv.imshow('frame', frame2)
            if cv.waitKey(1) == 27:
                # Esc key pressed. break
                break
            time.sleep(1 / fps)
            if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) < 1:
                # window closed. break
                break