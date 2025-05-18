import os, cv2, glob, random
import numpy as np
from pathlib import Path

# 경로 설정


output_img_dir = "/home/aistore02/Datasets/Augmented_data/img"
output_lbl_dir = "/home/aistore02/Datasets/Augmented_data/labels"
img_path = os.path.join(output_img_dir, "aff_composite_0923.png")
label_path = os.path.join(output_lbl_dir, "aff_composite_0923.txt")  # YOLO txt 파일 이름 맞춰서 수정


# output_img_dir = "/home/aistore02/Datasets/Augmented/cam0/img"
# output_lbl_dir = "/home/aistore02/Datasets/Augmented/cam0/labels"
# img_path = os.path.join(output_img_dir, "composite_0009.png")
# label_path = os.path.join(output_lbl_dir, "composite_0009.txt")  # YOLO txt 파일 이름 맞춰서 수정

# 사각형 + 라벨 표시 함수
def draw_bbox_with_label(image, label, color=(0, 255, 0), thickness=2):
    cls_id, cx, cy, w, h = label

    H, W = image.shape[:2]

    # YOLO -> pixel 좌표 변환
    px = int((cx - w / 2) * W)
    py = int((cy - h / 2) * H)
    pw = int(w * W)
    ph = int(h * H)

    # 사각형
    cv2.rectangle(image, (px, py), (px + pw, py + ph), color, thickness)

    # 클래스 id를 문자열로 라벨 표시
    label_str = str(cls_id)
    (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (px, py - text_h - 5), (px + text_w, py), color, -1)
    cv2.putText(image, label_str, (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    return image

# 이미지 불러오기
img = cv2.imread(img_path)

# 라벨 파일 읽기
labels = []
with open(label_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        label = [int(parts[0])] + [float(x) for x in parts[1:]]
        labels.append(label)

# 모든 바운딩 박스 그리기
for label in labels:
    draw_bbox_with_label(img, label)

# 결과 이미지 저장
cv2.imwrite("annotated_composite_0000.png", img)
