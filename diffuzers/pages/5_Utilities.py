import streamlit as st

from diffuzers.gfp_gan import GFPGAN
from diffuzers.image_info import ImageInfo
from diffuzers.upscaler import Upscaler


def app():
    task = st.selectbox(
        "Choose a utility",
        [
            "ImageInfo",
            "SD Upscaler",
            "GFPGAN",
        ],
    )
    # add a line break
    st.markdown("<hr>", unsafe_allow_html=True)
    if task == "ImageInfo":
        ImageInfo().app()
    elif task == "SD Upscaler":
        with st.form("upscaler_model"):
            upscaler_model = st.text_input("Model", "stabilityai/stable-diffusion-x4-upscaler")
            submit = st.form_submit_button("Load model")
        if submit:
            with st.spinner("Loading model..."):
                ups = Upscaler(
                    model=upscaler_model,
                    device=st.session_state.device,
                    output_path=st.session_state.output_path,
                )
                st.session_state.ups = ups
        if "ups" in st.session_state:
            st.write(f"Current model: {st.session_state.ups}")
            st.session_state.ups.app()

    elif task == "GFPGAN":
        with st.spinner("Loading model..."):
            gfpgan = GFPGAN(
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
        gfpgan.app()


if __name__ == "__main__":
    app()
