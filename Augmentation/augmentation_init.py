from utils.augmentation_utils import hsv_augment, affine_augment, cutout
import os
import glob
from pathlib import Path
import cv2

root_dir = os.path.expanduser("/home/aistore02/Datasets/Synthetic_data")
output_dir = os.path.expanduser("/home/aistore02/Datasets/Competition_Augmented_data")



for cam_idx in range(5):
    print(f"=== CAM{cam_idx} 데이터 생성 시작 ===")
    img_dir = os.path.join(root_dir, f"cam{cam_idx}", "img")
    label_dir = os.path.join(root_dir, f"cam{cam_idx}", "labels")

    img_paths = glob.glob(os.path.join(img_dir, "*.png")) + \
            glob.glob(os.path.join(img_dir, "*.jpg")) + \
            glob.glob(os.path.join(img_dir, "*.jpeg"))

    for img_path in img_paths:
        stem = Path(img_path).stem
        print(stem)
        label_path = os.path.join(label_dir, f"{stem}.txt")
        if not os.path.exists(label_path):
            print(f"라벨 없음: {label_path}")
            continue

        print(f"처리 중: {img_path}")
        image = cv2.imread(img_path)

        #아핀만 다 주석처리해놓음

        hsv_img = hsv_augment(image)
        #aff_img, aff_labels = affine_augment(image, label_path)
        cutout_img = cutout(image)

        #cv2.imwrite(os.path.join(output_dir, "img", f"hsv_{stem}.png"), hsv_img)
        #cv2.imwrite(os.path.join(output_dir, "img", f"aff_{stem}.png"), aff_img)
        cv2.imwrite(os.path.join(output_dir, "img", f"cutout_{stem}.png"), cutout_img)

        # with open(os.path.join(output_dir, "labels", f"aff_{stem}.txt"), "w") as f:
        #     for label in aff_labels:
        #         cls_id = int(label[0])
        #         coords = label[1:]
        #         f.write(f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords) + "\n")

                
        # HSV 라벨 복사
        # with open(label_path, 'r') as f_in, \
        #     open(os.path.join(output_dir, "labels", f"hsv_{stem}.txt"), "w") as f_out:
        #     f_out.write(f_in.read())

        # Cutout 라벨 복사
        with open(label_path, 'r') as f_in, \
            open(os.path.join(output_dir, "labels", f"cutout_{stem}.txt"), "w") as f_out:
            f_out.write(f_in.read())