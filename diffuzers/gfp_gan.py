import base64
import os
import shutil
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from loguru import logger
from PIL import Image
from realesrgan.utils import RealESRGANer

from diffuzers import utils


@dataclass
class GFPGAN:
    device: Optional[str] = None
    output_path: Optional[str] = None

    def __str__(self) -> str:
        return f"GFPGAN(device={self.device}, output_path={self.output_path})"

    def __post_init__(self):
        files = {
            "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "v1.2": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
            "v1.3": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            "v1.4": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            "RestoreFormer": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
            "CodeFormer": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth",
        }
        _ = utils.cache_folder()
        self.model_paths = {}
        for file_key, file in files.items():
            logger.info(f"Downloading {file_key} from {file}")
            basename = os.path.basename(file)
            output_path = os.path.join(utils.cache_folder(), basename)
            if os.path.exists(output_path):
                self.model_paths[file_key] = output_path
                continue
            temp_file = utils.download_file(file)
            shutil.move(temp_file, output_path)
            self.model_paths[file_key] = output_path

        self.model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        model_path = os.path.join(utils.cache_folder(), self.model_paths["realesr-general-x4v3.pth"])
        half = True if torch.cuda.is_available() else False
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=self.model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
        )

    def inference(self, img, version, scale):
        # taken from: https://huggingface.co/spaces/Xintao/GFPGAN/blob/main/app.py
        # weight /= 100
        if scale > 4:
            scale = 4  # avoid too large scale value

        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:  # for gray inputs
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        if version == "v1.2":
            face_enhancer = GFPGANer(
                model_path=self.model_paths["v1.2"],
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.upsampler,
            )
        elif version == "v1.3":
            face_enhancer = GFPGANer(
                model_path=self.model_paths["v1.3"],
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.upsampler,
            )
        elif version == "v1.4":
            face_enhancer = GFPGANer(
                model_path=self.model_paths["v1.4"],
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.upsampler,
            )
        elif version == "RestoreFormer":
            face_enhancer = GFPGANer(
                model_path=self.model_paths["RestoreFormer"],
                upscale=2,
                arch="RestoreFormer",
                channel_multiplier=2,
                bg_upsampler=self.upsampler,
            )
        # elif version == 'CodeFormer':
        #     face_enhancer = GFPGANer(
        #     model_path='CodeFormer.pth', upscale=2, arch='CodeFormer', channel_multiplier=2, bg_upsampler=upsampler)
        try:
            # _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        except RuntimeError as error:
            logger.error("Error", error)

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            logger.error("wrong scale input.", error)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output

    def app(self):
        input_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if input_image is not None:
            st.image(input_image, use_column_width=True)
        with st.form(key="gfpgan"):
            version = st.selectbox("GFPGAN version", ["v1.2", "v1.3", "v1.4", "RestoreFormer"])
            scale = st.slider("Scale", 2, 4, 4, 1)
            submit = st.form_submit_button("Upscale")
        if submit:
            if input_image is not None:
                with st.spinner("Upscaling image..."):
                    output_img = self.inference(input_image, version, scale)
                    st.image(output_img, use_column_width=True)
                    # add image download button
                    output_img = Image.fromarray(output_img)
                    buffered = BytesIO()
                    output_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    href = f'<a href="data:file/png;base64,{img_str}" download="gfpgan.png">Download Image</a>'
                    st.markdown(href, unsafe_allow_html=True)
