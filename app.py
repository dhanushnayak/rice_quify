import base64
from fastapi import FastAPI, File, UploadFile,Response
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
import uvicorn
import database
from image_analysis import get_analysis
import datetime
import time

IMAGEDIR = "images_data/"
try:
        os.path.exists(IMAGEDIR)
except:
        os.makedirs(IMAGEDIR)

app = FastAPI()

version = "1.1"


@app.get('/')
def index():
    return {"Version":version,"title":"RiceQ"}


@app.get('/images/name')
def get_images_name():
    names = database.get_images_name()
    return {"filename":names}

@app.post("/images/analysis/")
async def image_report(file: UploadFile = File(...)):
    global version
    file.filename = f"PIC_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg"
    contents = await file.read()  # <-- Important!

    # example of how you can save the file

    path = f"{IMAGEDIR}{file.filename}"
    with open(path, "wb") as f:
        f.write(contents)
    data = get_analysis(path)
    while True:
        res = database.insertBLOB(file.filename,path)
        time.sleep(0.2)
        if res==True:
            os.remove(path)
            print("INSERTED DELETED FROM TEMP")
            break
        
    
    return {**{"filename": file.filename},**data,"version":version}


@app.get("/images/{filename}")
async def get_images(filename):

    res = database.Get_Image(filename)[1]
    # notice you can use FileResponse now because it expects a path
    return Response(content=res, media_type="image/png")


