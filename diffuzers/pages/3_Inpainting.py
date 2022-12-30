import streamlit as st

from diffuzers.inpainting import Inpainting


def app():
    with st.form("inpainting_model"):
        model = st.text_input("Which model do you want to use?", value="runwayml/stable-diffusion-inpainting")
        submit = st.form_submit_button("Load model")
    if submit:
        with st.spinner("Loading model..."):
            inpainting = Inpainting(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.inpainting = inpainting
    if "inpainting" in st.session_state:
        st.write(f"Current model: {st.session_state.inpainting}")
        st.session_state.inpainting.app()


if __name__ == "__main__":
    app()
