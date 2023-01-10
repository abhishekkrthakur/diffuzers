import io
import os

from fastapi import Depends, FastAPI, File, UploadFile
from loguru import logger
from PIL import Image
from starlette.middleware.cors import CORSMiddleware

from diffuzers.api.schemas import Img2ImgParams, ImgResponse, InpaintingParams, Text2ImgParams
from diffuzers.api.utils import convert_to_b64_list
from diffuzers.inpainting import Inpainting
from diffuzers.x2image import X2Image


app = FastAPI(
    title="diffuzers api",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup_event():

    x2img_model = os.environ.get("X2IMG_MODEL")
    x2img_pipeline = os.environ.get("X2IMG_PIPELINE")
    inpainting_model = os.environ.get("INPAINTING_MODEL")
    device = os.environ.get("DEVICE")
    output_path = os.environ.get("OUTPUT_PATH")
    ti_identifier = os.environ.get("TOKEN_IDENTIFIER", "")
    ti_embeddings_url = os.environ.get("TOKEN_EMBEDDINGS_URL", "")
    logger.info("@@@@@ Starting Diffuzes API @@@@@ ")
    logger.info(f"Text2Image Model: {x2img_model}")
    logger.info(f"Text2Image Pipeline: {x2img_pipeline if x2img_pipeline is not None else 'Vanilla'}")
    logger.info(f"Inpainting Model: {inpainting_model}")
    logger.info(f"Device: {device}")
    logger.info(f"Output Path: {output_path}")
    logger.info(f"Token Identifier: {ti_identifier}")
    logger.info(f"Token Embeddings URL: {ti_embeddings_url}")

    logger.info("Loading x2img model...")
    if x2img_model is not None:
        app.state.x2img_model = X2Image(
            model=x2img_model,
            device=device,
            output_path=output_path,
            custom_pipeline=x2img_pipeline,
            token_identifier=ti_identifier,
            embeddings_url=ti_embeddings_url,
        )
    else:
        app.state.x2img_model = None
    logger.info("Loading inpainting model...")
    if inpainting_model is not None:
        app.state.inpainting_model = Inpainting(
            model=inpainting_model,
            device=device,
            output_path=output_path,
        )
    logger.info("API is ready to use!")


@app.post("/text2img")
async def text2img(params: Text2ImgParams) -> ImgResponse:
    logger.info(f"Params: {params}")
    if app.state.x2img_model is None:
        return {"error": "x2img model is not loaded"}
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
    return ImgResponse(images=base64images, metadata=params.dict())


@app.post("/img2img")
async def img2img(params: Img2ImgParams = Depends(), image: UploadFile = File(...)) -> ImgResponse:
    if app.state.x2img_model is None:
        return {"error": "x2img model is not loaded"}
    image = Image.open(io.BytesIO(image.file.read()))
    images, _ = app.state.x2img_model.img2img_generate(
        image=image,
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        num_images=params.num_images,
        steps=params.steps,
        seed=params.seed,
        scheduler=params.scheduler,
        guidance_scale=params.guidance_scale,
        strength=params.strength,
    )
    base64images = convert_to_b64_list(images)
    return ImgResponse(images=base64images, metadata=params.dict())


@app.post("/inpainting")
async def inpainting(
    params: InpaintingParams = Depends(), image: UploadFile = File(...), mask: UploadFile = File(...)
) -> ImgResponse:
    if app.state.inpainting_model is None:
        return {"error": "inpainting model is not loaded"}
    image = Image.open(io.BytesIO(image.file.read()))
    mask = Image.open(io.BytesIO(mask.file.read()))
    images, _ = app.state.inpainting_model.generate_image(
        image=image,
        mask=mask,
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        scheduler=params.scheduler,
        height=params.image_height,
        width=params.image_width,
        num_images=params.num_images,
        guidance_scale=params.guidance_scale,
        steps=params.steps,
        seed=params.seed,
    )
    base64images = convert_to_b64_list(images)
    return ImgResponse(images=base64images, metadata=params.dict())


@app.get("/")
def read_root():
    return {"Hello": "World"}
