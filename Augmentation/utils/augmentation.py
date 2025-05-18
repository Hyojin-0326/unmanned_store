import cv2
import numpy as np
import random
import albumentations as A

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
    for label in labels:
        bbox, cls_id = yolo_to_voc(label)
        voc_bboxes.append(bbox)
        class_ids.append(cls_id)

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

def cutout(img, num_holes=1, max_h_size=60, max_w_size=60):
    h, w = img.shape[:2]
    new_img = img.copy()

    for _ in range(num_holes):
        hole_w = random.randint(1, max_w_size)
        hole_h = random.randint(1, max_h_size)
        x = random.randint(0, max(1, w-hole_w))
        y = random.randint(0, max(1, h-hole_h))

        region = new_img[y:y+hole_h, x:x+hole_w]
        fillValue = region.mean(axis=(0, 1)).astype(np.uint8)

        new_img[y:y+hole_h, x:x+hole_w] = fillValue
    return new_img


