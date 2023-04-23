from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

ims = [Image.open("img1.JPG"), Image.open("img4.JPG"), Image.open("img10.JPG"), Image.open("img11.jpg"),
       Image.open("img5.JPG"), Image.open("img6.JPG"), Image.open("img7.JPG"), Image.open("img8.JPG"), 
       Image.open("img9.JPG")]

# results = model.predict(source=ims)
results = model(ims)

for i, res in enumerate(results):
    res_plotted = res.plot()
    cv2.imwrite(f'result{i}.jpg', res_plotted)

