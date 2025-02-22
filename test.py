import torch
from ultralytics import YOLO
import glob
import os

model = YOLO(r"D:\TheCodingBug\my_model.pt")
    

image_folder = r"D:\TheCodingBug\yolov_11_custom\val\images"


image_paths = glob.glob(image_folder + r"\*.jpg") + glob.glob(image_folder + r"\*.png")


output_folder = r"D:\TheCodingBug\yolov_11_custom\results"


if os.path.exists(output_folder):
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)  


for image_path in image_paths:
    print(f"Processing: {image_path}")
    results = model(image_path)  

   
    for i, result in enumerate(results):
        filename = os.path.basename(image_path)  
        save_path = os.path.join(output_folder, filename)  
        result.save(filename=save_path) 

print("✅ Detection completed. Check the 'results/' folder for output images.")


