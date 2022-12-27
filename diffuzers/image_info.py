from dataclasses import dataclass

import streamlit as st
from PIL import Image


@dataclass
class ImageInfo:
    def app(self):
        # upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # read image using pil
            pil_image = Image.open(uploaded_file)
            st.image(uploaded_file, use_column_width=True)
            image_info = pil_image.info
            # display image info
            st.write(image_info)
