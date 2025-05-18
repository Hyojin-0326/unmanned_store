# generate_all_cams.py
import os
import glob
from git.Augmentation.utils.synthesis_utils import compose_one

root_dir = os.path.expanduser("~/Datasets/3.backsub_images_100")
base_bg_dir = "/home/aistore02/Datasets/3.Background_Images"
base_out_dir = "/home/aistore02/Datasets/Augmented"

# cam0 ~ cam5 처리
for cam_idx in range(5):
    print(f"=== CAM{cam_idx} 데이터 생성 시작 ===")
    background_dir = os.path.join(base_bg_dir, f"cam{cam_idx}")
    output_img_dir = os.path.join(base_out_dir, f"cam{cam_idx}", "img")
    output_lbl_dir = os.path.join(base_out_dir, f"cam{cam_idx}", "labels")

    bg_paths = glob.glob(os.path.join(background_dir, "*.png"))
    class_names = sorted(os.listdir(root_dir))

    for i in range(20000):
        compose_one(i, root_dir, background_dir, output_img_dir, output_lbl_dir, bg_paths, class_names)
