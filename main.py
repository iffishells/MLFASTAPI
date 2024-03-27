from fastapi import FastAPI
from pydantic import BaseModel
from ml import obtain_image
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse

import io
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get('generate')
def generate_image(prompt: str):
    """

    :param prompt:
    :return:
    """
    image = obtain_image(prompt,
                         num_inference_steps=50,
                         guidance_scale=7.5)
    memory_stream = io.BytesIO()
    image.save(memory_stream,format='PNG')
    memory_stream.seek(0)
    return StreamingResponse(memory_stream,media_type="image/png")
