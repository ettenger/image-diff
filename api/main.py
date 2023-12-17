from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from image_diff import find_diff

app = FastAPI()

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS","HEAD","PUT","PATCH","DELETE"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/check-diff/")
async def check_diff(reference_image: UploadFile, given_image: UploadFile):
    return find_diff(reference_image.file, given_image.file)
