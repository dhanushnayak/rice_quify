from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
import uvicorn

from image_analysis import get_analysis

IMAGEDIR = "images_data/"

app = FastAPI()

version = "1.0"

@app.post("/images/analysis/")
async def create_upload_file(file: UploadFile = File(...)):
    global version
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  # <-- Important!

    # example of how you can save the file
    path = f"{IMAGEDIR}{file.filename}"
    with open(path, "wb") as f:
        f.write(contents)
    data = get_analysis(path)
    return {**{"filename": file.filename},**data,"version":version}


@app.get("/images/{filename}")
async def read_random_file(filename):

    # get a random file from the image directory

    path = f"{IMAGEDIR}{filename}"
    
    # notice you can use FileResponse now because it expects a path
    return FileResponse(path)


