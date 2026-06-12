import os
import cv2
import xml.etree.ElementTree as ET

IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"

HELMET_DIR = "helmet"
NO_HELMET_DIR = "no_helmet"

os.makedirs(HELMET_DIR, exist_ok=True)
os.makedirs(NO_HELMET_DIR, exist_ok=True)

helmet_count = 0
nohelmet_count = 0

PADDING = 30

for xml_file in os.listdir(ANNOTATION_DIR):

    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATION_DIR, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(IMAGE_DIR, image_name)

    img = cv2.imread(image_path)

    if img is None:
        continue

    h, w = img.shape[:2]

    for obj in root.findall("object"):

        label = obj.find("name").text.strip()

        box = obj.find("bndbox")

        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)

        xmin = max(0, xmin - PADDING)
        ymin = max(0, ymin - PADDING)
        xmax = min(w, xmax + PADDING)
        ymax = min(h, ymax + PADDING)

        crop = img[ymin:ymax, xmin:xmax]

        if crop.size == 0:
            continue

        if label == "With Helmet":

            save_path = os.path.join(
                HELMET_DIR,
                f"helmet_{helmet_count}.jpg"
            )

            cv2.imwrite(save_path, crop)
            helmet_count += 1

        elif label == "Without Helmet":

            save_path = os.path.join(
                NO_HELMET_DIR,
                f"nohelmet_{nohelmet_count}.jpg"
            )

            cv2.imwrite(save_path, crop)
            nohelmet_count += 1

print(f"Helmet images: {helmet_count}")
print(f"No helmet images: {nohelmet_count}")
print("Done!")