import base64
import io

from diffuzers.x2image import X2Image


def x2img_model(model, pipeline, device, ti_identifier, ti_embeddings_url, output_path):
    x2img = X2Image(
        model=model,
        device=device,
        output_path=output_path,
        custom_pipeline=pipeline,
        token_identifier=ti_identifier,
        embeddings_url=ti_embeddings_url,
    )
    return x2img


def convert_to_b64_list(images):
    base64images = []
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = base64.b64encode(buf.getvalue())
        base64images.append(byte_im)
    return base64images
