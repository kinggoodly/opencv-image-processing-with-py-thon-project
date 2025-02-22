import os
import sys
import cv2
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import unicodedata  # To normalize Thai text
from difflib import get_close_matches  # For closest match detection

# Load Thai font
font_path = "D:\TheCodingBug\yolov_11_custom\THSarabunNew.ttf"
if not os.path.exists(font_path):
    print("❌ Thai Font not found! Check the path.")
    sys.exit()
font = ImageFont.truetype(font_path, 32)

# Define paths and variables
model_path = r"D:\TheCodingBug\my_model.pt"
min_thresh = 0.50
cam_index = 0
imgW, imgH = 1280, 720

# Normalize Thai text to prevent mismatches
def normalize_text(text):
    return unicodedata.normalize('NFKC', text).strip()

# Product price dictionary with normalized keys
nutrition_info = {
    normalize_text('น้ำดื่มคริสตัล'): 7, 
    normalize_text('เลย์คลาสสิครสโนริ'): 32, 
    normalize_text('โค้ก 325 มล.'): 16,  
    normalize_text('เยลลี่จอลลี่เเบร์'): 10, 
    normalize_text('ลูกอมโมรินากะไฮชิว'): 20
}

# Function to find closest match in case of minor OCR mismatches
def find_closest_match(item_name, data_dict):
    matches = get_close_matches(item_name, data_dict.keys(), n=1, cutoff=0.8)
    return matches[0] if matches else None

# Load YOLO model
if not os.path.exists(model_path):
    print('⚠️ Model not found.')
    sys.exit()

model = YOLO(model_path, task='detect')

# Normalize class names in YOLO model
labels = {idx: normalize_text(name) for idx, name in model.names.items()}

# Initialize camera
cap = cv2.VideoCapture(cam_index)
cap.set(3, imgW)
cap.set(4, imgH)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error.")
        break

    results = model.track(frame, verbose=False)
    detections = results[0].boxes
    item_detected = []

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]  # Already normalized

        conf = detections[i].conf.item()
        if conf > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Convert OpenCV image to PIL for Thai text support
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)

            text = f"{classname}: {int(conf*100)}%"
            draw.text((xmin, ymin - 30), text, font=font, fill=(0, 255, 0))

            # Convert back to OpenCV
            frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Add detected item to list
            item_detected.append(classname)

    # Calculate total price
    total_price = 0
    for item_name in item_detected:
        print(f"🔍 Detected: '{item_name}'")  # Debugging print
        if item_name in nutrition_info:
            total_price += nutrition_info[item_name]
        else:
            closest_match = find_closest_match(item_name, nutrition_info)
            if closest_match:
                total_price += nutrition_info[closest_match]
                print(f"✅ Using closest match: '{closest_match}' instead of '{item_name}'")
            else:
                print(f"⚠️ '{item_name}' not found in database")

    # Display total price on frame
    cv2.rectangle(frame, (10, 10), (450, 130), (50, 50, 50), cv2.FILLED)
    cv2.putText(frame, f'Items: {len(item_detected)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 102, 51), 2)
    cv2.putText(frame, f'Total Price: {total_price} bath', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 204, 51), 2)

    # Show result
    cv2.imshow('Item detection results', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
