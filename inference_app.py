from fastapi import FastAPI, File, UploadFile
from inference import InferenceOCR
import numpy as np
from PIL import Image
from fastapi_utils.tasks import repeat_every
import shutil


app = FastAPI()
inference_ocr = InferenceOCR()


@app.post("/set-model")
async def set_model(model_path: str):
    inference_ocr.set_model(model_path)
    return {"status": "success", "message": "model set to {}".format(model_path)}


@app.post("/set-batch")
async def set_batch(batch_size: int):
    inference_ocr.batch_size = batch_size
    return {"status": "success", "message": "batch size set to {}".format(batch_size)}


@app.post("/inference")
async def inference(images: list):
    images_path = []
    results = []
    for image in images:
        pil_image = Image.open(image.file)
        np_image = np.array(pil_image)
        images_path.append(np_image)
        res = inference_ocr.inference(images_path)
        results.append(res)
    return results


@app.on_event("startup")
@repeat_every(seconds=15)
async def repeat_every_5_seconds():
    shutil.rmtree('lightning_logs')
