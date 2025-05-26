import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO

def extract_license_plates_from_coco(image_dir, annotation_file):
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    X = []
    y = []

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        ann_ids = coco.getAnnIds(imgIds=img_info['id'])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            x, y_, w, h = map(int, ann['bbox'])
            crop = img[y_:y_+h, x:x+w]
            if crop.size == 0:
                continue

            # Resize para normalizar (por ejemplo, 64x64)
            crop_resized = cv2.resize(crop, (64, 64))
            gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            X.append(gray.flatten())
            y.append(ann['category_id'])  # Cambiar si es multiclase o binario

    return np.array(X), np.array(y)