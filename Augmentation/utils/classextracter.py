import os

folder = "/home/aistore02/Datasets/60class_zoom"  # <- 이미지 있는 폴더 경로

class_names = []
for fname in sorted(os.listdir(folder)):
    if fname.endswith(".jpg"):
        name = fname.split(".", 1)[1].rsplit(".", 1)[0]  # '0.abc.jpg' → 'abc'
        class_names.append(name)

print(class_names)
