import cv2
import numpy as np
import random
import albumentations as A
import os

def hsv_augment(image, hue_shift=10, sat_shift=20, val_shift=10): #sat_shift, val_shift: 0~255 범위 내 조절
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32))
    h=(h+random.randint(-hue_shift, hue_shift))%180
    s=np.clip(s+random.randint(-sat_shift, sat_shift), 0, 255)
    v= np.clip(v+random.randint(-val_shift, val_shift), 0, 255)

    transformed = cv2.merge([h,s,v]).astype(np.uint8)
    return transformed

def affine_augment(image,label_path, max_translate = 0.1, max_rotate=10, max_scale=0.2):
    h, w = image.shape[:2]
      # YOLO → Pascal VOC 변환
    def yolo_to_voc(bbox):
        cls, cx, cy, bw, bh = bbox
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        return [x1, y1, x2, y2], int(cls)

    voc_bboxes = []
    class_ids = []

    with open(label_path, 'r') as f:
        for line in f:
            comps = line.strip().split()
            if len(comps) != 5:
                continue
            bbox = [float(x) for x in comps]
            voc_bbox, cls = yolo_to_voc(bbox)
            voc_bboxes.append(voc_bbox)
            class_ids.append(cls)

    # Affine transform 정의
    transform = A.Compose([
        A.ShiftScaleRotate(
            shift_limit=max_translate,
            scale_limit=max_scale,
            rotate_limit=max_rotate,
            border_mode=cv2.BORDER_REFLECT101,
            p=1.0
        ),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    augmented = transform(image=image, bboxes=voc_bboxes, class_labels=class_ids)
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_class_ids = augmented['class_labels']

    # Pascal VOC → YOLO 포맷 복원
    def voc_to_yolo(x1, y1, x2, y2, cls_id):
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2
        cy = y1 + bh / 2
        return [cls_id, cx / w, cy / h, bw / w, bh / h]

    final_labels = [voc_to_yolo(*bbox, cls) for bbox, cls in zip(aug_bboxes, aug_class_ids)]

    return aug_image, final_labels

def cutout(img, label_path=None, min_num_holes = 3, max_num_holes=8, max_h_size=100, max_w_size=100,
           fill_mode='noise', min_iou_with_bbox=0.2, max_iou_with_bbox=0.5):
    h, w = img.shape[:2]
    new_img = img.copy()
    # bbox 불러오기 (YOLO 형식)
    bboxes = []
    if label_path and os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                comps = list(map(float, line.strip().split()))
                if len(comps) == 5:
                    cls, cx, cy, bw, bh = comps
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    bboxes.append([x1, y1, x2, y2])

    def compute_iou(box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        if inter_area == 0:
            return 0.0

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / float(box1_area + box2_area - inter_area)

    num_holes = np.random.randint(min_num_holes, max_num_holes + 1)
    count = 0
    trials = 0
    while count < num_holes and trials < 100:
        hole_w = random.randint(20, max_w_size)
        hole_h = random.randint(20, max_h_size)
        x1 = random.randint(0, max(1, w - hole_w))
        y1 = random.randint(0, max(1, h - hole_h))
        x2 = x1 + hole_w
        y2 = y1 + hole_h

        ious = [compute_iou([x1, y1, x2, y2], bb) for bb in bboxes]
        if any(min_iou_with_bbox <= iou <= max_iou_with_bbox for iou in ious):
            region = new_img[y1:y2, x1:x2]
            if fill_mode == 'mean':
                fillValue = region.mean(axis=(0, 1)).astype(np.uint8)
            elif fill_mode == 'noise':
                fillValue = np.random.randint(0, 256, size=(hole_h, hole_w, 3), dtype=np.uint8)

            new_img[y1:y2, x1:x2] = fillValue
            count += 1

        trials += 1

    return new_img

