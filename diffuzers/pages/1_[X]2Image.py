import streamlit as st

from diffuzers.x2image import X2Image


def app():
    with st.form("x2img_model_form"):
        model = st.text_input(
            "Which model do you want to use?",
            value="stabilityai/stable-diffusion-2-base"
            if st.session_state.get("x2img_model") is None
            else st.session_state.x2img_model,
        )
        custom_pipeline = st.selectbox(
            "Custom pipeline",
            options=[
                "Vanilla",
                "Long Prompt Weighting",
            ],
            index=0 if st.session_state.get("x2img_custom_pipeline") == "Vanilla" else 1,
        )
        submit = st.form_submit_button("Load model")

    if submit:
        st.session_state.x2img_model = model
        st.session_state.x2img_custom_pipeline = custom_pipeline
        custom_pipeline = "lpw_stable_diffusion" if custom_pipeline == "Long Prompt Weighting" else None
        with st.spinner("Loading model..."):
            x2img = X2Image(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
                custom_pipeline=custom_pipeline,
            )
            st.session_state.x2img = x2img
    if "x2img" in st.session_state:
        st.write(f"Current model: {st.session_state.x2img}")
        st.session_state.x2img.app()


if __name__ == "__main__":
    app()
