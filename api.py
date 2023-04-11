import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from typing import Literal, List, Union
import cv2
from PIL import Image
from ultralytics import YOLO
import shutil



description = """
Welcome to Wildfire_Project api 

## Introduction endpoint

* `/`: **GET** request that display a simple default message. 

## Detector endpoint

* `/image-detector`: **POST** request that runs the Wildfire detection model on uploaded image and outputs the result. 

Check out documentation below 👇 for more information on each endpoint. 
"""

tags_metadata = [
    {
        "name": "Introduction Endpoint",
        "description": "Simple endpoint to try out"
    },

    {
        "name": "Detector Endpoint",
        "description": "Wildfire Predictor"
    }
]


app = FastAPI(
    title="🔥 Wildfire Project API",
    description=description,
    version="0.1",
    contact={
        "name": "Wildfire",
        "url": "https://wildfire-streamlit.herokuapp.com/",
    },
    openapi_tags=tags_metadata
)



@app.get("/", tags= ["Introduction Endpoint"])
async def index():
    """
    Simply returns an intro message

    """
    message = "Hello world! Please use POST request to use the wildfire_detector model. For more info go to /docs"

    return message

@app.post("/image-detector", tags= ["Detector Endpoint"])
async def image_pred(file: UploadFile = File(...)):
    """
    Make image prediction 
    """ 
    # Clear previously detected images folder (need to add condition if folder exist)
    #shutil.rmtree('runs/detect/')

    # Load model
    model = YOLO("yolov8_run2.pt")

    # Run trained model on uploaded image.
    res = model(Image.open(file.file), save=True)

    return FileResponse(f"runs/detect/predict/{file.filename}") 
    




if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4001) # Here you define your web server to run the `app` variable (which contains FastAPI instance), with a specific host IP (0.0.0.0) and port (4000)

