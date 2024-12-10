import datetime
import io
import os

from fastapi import FastAPI, File, UploadFile, Form
from typing import Annotated
import uvicorn
import torch
from torchvision import transforms
# from torchvision.models import EfficientNet, efficientnet_b0
from PIL import Image
from pathlib import Path

app = FastAPI(title="Animals Hangman API")

TRAINING_REQUESTS_PATH = Path("train-requests")


def load_model():
    transfer_model = torch.load("transfer_cnn_model.pth", map_location=torch.device("cpu"))
    transfer_model.eval()
    return transfer_model


model = load_model()
class_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken',
               'cat', 'cow', 'sheep', 'spider', 'squirrel']


@app.get("/")
def home():
    return "Welcome to Animals Hangman API"


@app.get("/classes")
def get_classes():
    return {"classes": class_names}


@app.post('/predict', status_code=200)
def predict(file: Annotated[bytes, File()]):
    img = Image.open(io.BytesIO(file))
    transformer = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(size=(64, 64)),
        transforms.ConvertImageDtype(torch.float),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize for EfficientNet
        #                      std=[0.229, 0.224, 0.225])
        # transforms.RandomHorizontalFlip(p=0.5)
    ])
    img_tensor = transformer(img)

    result = model(img_tensor.unsqueeze(dim=0))
    probabilities = result.softmax(dim=-1)
    prediction = class_names[probabilities.argmax(dim=-1)]
    class_probabilities = dict(zip(class_names, probabilities.squeeze().tolist()))
    class_probabilities = dict(sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True))
    return {"prediction": prediction, "probabilities": class_probabilities}


@app.post('/train-request', status_code=201)
async def request_train(class_name: Annotated[str, Form()],
                        file: Annotated[UploadFile, File()]):
    file_content = await file.read()
    file_extension = Path(file.filename).suffix.lstrip('.')
    class_name = class_name.lower()

    image_class_path = TRAINING_REQUESTS_PATH / class_name
    image_class_path.mkdir(parents=True, exist_ok=True)

    file_id = len(os.listdir(image_class_path)) + 1
    date = str(datetime.date.today())
    file_name = f"{class_name}-{file_id}_{date}.{file_extension}"
    image_full_path = image_class_path / file_name

    with open(image_full_path, 'wb') as f:
        f.write(file_content)


@app.get("/train-requests")
def get_training_requests():
    directories = dict()
    for dir_path, dir_names, file_names in os.walk(TRAINING_REQUESTS_PATH):
        if dir_path == TRAINING_REQUESTS_PATH.name:
            continue
        directories[os.path.basename(dir_path)] = len(file_names)
    return directories


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
