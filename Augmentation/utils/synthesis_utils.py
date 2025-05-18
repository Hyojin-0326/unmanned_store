import os, cv2, glob, random
import numpy as np


def convert_bbox(x, y, w, h, img_w, img_h):
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def compose_one(idx, root_dir, background_dir, output_img_dir, output_lbl_dir, bg_paths, class_names):
    bg_path = random.choice(bg_paths)
    bg = cv2.imread(bg_path)
    if bg is None:
        print(f"[{idx}] 배경 이미지 로딩 실패: {bg_path}")
        return

    h_bg, w_bg = bg.shape[:2]
    label_lines = []
    num_objs = random.randint(2, 12)

    for _ in range(num_objs):
        cls_name = random.choice(class_names)
        cls_dir = os.path.join(root_dir, cls_name)
        img_files = glob.glob(os.path.join(cls_dir, "*.jpg")) + \
                    glob.glob(os.path.join(cls_dir, "*.png")) + \
                    glob.glob(os.path.join(cls_dir, "*.jpeg"))

        if not img_files:
            continue

        img_path = random.choice(img_files)
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            continue

        obj_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if obj_img is None or obj_img.shape[2] < 3:
            continue

        scale = random.uniform(0.4, 0.7)
        new_w = int(obj_img.shape[1] * scale)
        new_h = int(obj_img.shape[0] * scale)
        obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        oh, ow = obj_img.shape[:2]
        if oh >= h_bg or ow >= w_bg:
            continue

        x = random.randint(0, w_bg - ow)
        y = random.randint(0, h_bg - oh)

        black_mask = cv2.inRange(obj_img, (0, 0, 0), (10, 10, 10))
        alpha = cv2.bitwise_not(black_mask) / 255.0
        alpha_expanded = alpha[:, :, None]

        obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2BGRA)
        obj_img[:, :, 3] = (alpha * 255).astype(np.uint8)

        bg[y:y+oh, x:x+ow, :3] = (
            (1 - alpha_expanded) * bg[y:y+oh, x:x+ow, :3] +
            alpha_expanded * obj_img[:, :, :3]
        )

        with open(txt_path, 'r') as f:
            for line in f:
                comps = line.strip().split()
                if len(comps) != 5:
                    continue
                cls_id, cx, cy, w, h = map(float, comps)
                abs_x = int(cx * ow)
                abs_y = int(cy * oh)
                abs_w = int(w * ow)
                abs_h = int(h * oh)

                new_x = x + abs_x - abs_w // 2
                new_y = y + abs_y - abs_h // 2
                new_bbox = convert_bbox(new_x, new_y, abs_w, abs_h, w_bg, h_bg)
                label_lines.append(f"{int(cls_id)} {' '.join(f'{v:.6f}' for v in new_bbox)}")

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)
    out_img_path = os.path.join(output_img_dir, f"composite_{idx:04d}.png")
    out_lbl_path = os.path.join(output_lbl_dir, f"composite_{idx:04d}.txt")
    cv2.imwrite(out_img_path, bg)
    with open(out_lbl_path, "w") as f:
        f.write("\n".join(label_lines))

    print(f"[{idx}] 저장 완료: {out_img_path}, 라벨 {len(label_lines)}개")
