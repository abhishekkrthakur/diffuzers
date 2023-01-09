import io

from fastapi import Depends, FastAPI, File, UploadFile
from loguru import logger
from PIL import Image
from starlette.middleware.cors import CORSMiddleware

from diffuzers.api.schemas import Img2ImgParams, Text2ImgParams, X2ImgResponse
from diffuzers.api.utils import convert_to_b64_list, x2img_model


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup_event():
    logger.info("Loading x2img model...")
    app.state.x2img_model = x2img_model(
        model="stabilityai/stable-diffusion-2-base",
        pipeline=None,
        device="cuda",
        ti_identifier="",
        ti_embeddings_url="",
        output_path=None,
    )


@app.post("/text2img")
def text2img(params: Text2ImgParams) -> X2ImgResponse:
    logger.info(f"Params: {params}")
    images, _ = app.state.x2img_model.text2img_generate(
        params.prompt,
        num_images=params.num_images,
        steps=params.steps,
        seed=params.seed,
        negative_prompt=params.negative_prompt,
        scheduler=params.scheduler,
        image_size=(params.image_height, params.image_width),
        guidance_scale=params.guidance_scale,
    )
    base64images = convert_to_b64_list(images)
    return X2ImgResponse(images=base64images, metadata=params.dict())


@app.post("/img2img")
def img2img(
    params: Img2ImgParams = Depends(), image: UploadFile = File(...), mask: UploadFile = File(...)
) -> X2ImgResponse:
    image = Image.open(io.BytesIO(image.file.read()))
    mask = Image.open(io.BytesIO(mask.file.read()))
    images, _ = app.state.x2img_model.img2img_generate(
        image=image,
        mask=mask,
        num_images=params.num_images,
        steps=params.steps,
        seed=params.seed,
        negative_prompt=params.negative_prompt,
        scheduler=params.scheduler,
        guidance_scale=params.guidance_scale,
        strength=params.strength,
    )
    base64images = convert_to_b64_list(images)
    return X2ImgResponse(images=base64images, metadata=params.dict())


@app.get("/")
def read_root():
    return {"Hello": "World"}
