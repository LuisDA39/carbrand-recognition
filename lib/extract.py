import os
import json
import cv2
import numpy as np
from collections import defaultdict
from skimage.feature import hog
from skimage.color import rgb2gray
from pycocotools.coco import COCO

def extract_brand_samples_from_coco(images_dir, annotation_path):
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Mapeo de ID a nombre de archivo
    image_id_to_file = {img['id']: img['file_name'] for img in data['images']}

    # Agrupar anotaciones por imagen
    image_annotations = defaultdict(list)
    for ann in data['annotations']:
        image_annotations[ann['image_id']].append(ann)

    X = []
    y = []

    for img_id, anns in image_annotations.items():
        file_name = image_id_to_file[img_id]
        full_path = os.path.join(images_dir, file_name)

        # Asumimos una sola marca por imagen
        class_id = anns[0]['category_id']

        try:
            features = extract_hog_features(full_path)
            X.append(features)
            y.append(class_id)
        except Exception as e:
            print(f"Error en {full_path}: {e}")

    return np.array(X), np.array(y)

def extract_hog_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    gray = rgb2gray(img)  # convertir a escala de grises

    features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    return features