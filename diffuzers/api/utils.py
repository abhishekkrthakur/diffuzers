import base64
import io


def convert_to_b64_list(images):
    base64images = []
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = base64.b64encode(buf.getvalue())
        base64images.append(byte_im)
    return base64images
