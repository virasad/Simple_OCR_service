from fastapi import FastAPI

from train import ocr_trainer

app = FastAPI(title="OCR API", description="OCR API this package developed for the purpose of OCR")


@app.post("/train")
async def train(img_w: int, img_h: int, labels_txt_p: str, images_path: str, checkpoint_path: str, model_name: str,
                max_epochs: int = 1000, lr: float = 1e-3, batch_size: int = 128, log_url: str = None,
                task_id: str = None):
    ocr_trainer(img_w=img_w, img_h=img_h,
                labels_txt_p=labels_txt_p,
                images_path=images_path,
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                max_epochs=max_epochs,
                lr=lr,
                batch_size=batch_size,
                log_url=log_url,
                task_id=task_id)
    return {"status": "success", "message": "model trained"}
