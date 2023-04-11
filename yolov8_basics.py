from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model or our custom model
model = YOLO(model="weights/best.pt")  

# predict on an image
detection_output = model.predict(source="inferences/images/GP1SU5J0_PressMedia.JPG", conf=0.25, save=True) 

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())