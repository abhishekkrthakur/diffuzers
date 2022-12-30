import streamlit as st

from diffuzers.img2img import Img2Img
from diffuzers.text2img import Text2Image


def app():
    if "text2img" in st.session_state and "img2img" not in st.session_state:
        img2img = Img2Img(
            model=None,
            device=st.session_state.device,
            output_path=st.session_state.output_path,
            text2img_model=st.session_state.text2img.pipeline,
        )
        st.session_state.img2img = img2img
    with st.form("img2img_model"):
        model = st.text_input("Which model do you want to use?", value="stabilityai/stable-diffusion-2-base")
        submit = st.form_submit_button("Load model")
    if submit:
        with st.spinner("Loading model..."):
            img2img = Img2Img(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.img2img = img2img
            text2img = Text2Image(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.text2img = text2img
    if "img2img" in st.session_state:
        st.write(f"Current model: {st.session_state.img2img}")
        st.session_state.img2img.app()


if __name__ == "__main__":
    app()
