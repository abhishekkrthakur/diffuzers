import argparse

import streamlit as st
from loguru import logger

from diffuzers import utils
from diffuzers.x2image import X2Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help="Output path",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to use, e.g. cpu, cuda, cuda:0, mps etc.",
    )
    return parser.parse_args()


def x2img_app():
    with st.form("x2img_model_form"):
        col1, col2 = st.columns(2)
        with col1:
            model = st.text_input(
                "Which model do you want to use?",
                value="stabilityai/stable-diffusion-2-base"
                if st.session_state.get("x2img_model") is None
                else st.session_state.x2img_model,
            )
        with col2:
            custom_pipeline = st.selectbox(
                "Custom pipeline",
                options=[
                    "Vanilla",
                    "Long Prompt Weighting",
                ],
                index=0 if st.session_state.get("x2img_custom_pipeline") in (None, "Vanilla") else 1,
            )

        with st.expander("Textual Inversion (Optional)"):
            token_identifier = st.text_input(
                "Token identifier",
                placeholder="<something>"
                if st.session_state.get("textual_inversion_token_identifier") is None
                else st.session_state.textual_inversion_token_identifier,
            )
            embeddings = st.text_input(
                "Embeddings",
                placeholder="https://huggingface.co/sd-concepts-library/axe-tattoo/resolve/main/learned_embeds.bin"
                if st.session_state.get("textual_inversion_embeddings") is None
                else st.session_state.textual_inversion_embeddings,
            )
        submit = st.form_submit_button("Load model")

    if submit:
        st.session_state.x2img_model = model
        st.session_state.x2img_custom_pipeline = custom_pipeline
        st.session_state.textual_inversion_token_identifier = token_identifier
        st.session_state.textual_inversion_embeddings = embeddings
        cpipe = "lpw_stable_diffusion" if custom_pipeline == "Long Prompt Weighting" else None
        with st.spinner("Loading model..."):
            x2img = X2Image(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
                custom_pipeline=cpipe,
                token_identifier=token_identifier,
                embeddings_url=embeddings,
            )
            st.session_state.x2img = x2img
    if "x2img" in st.session_state:
        st.write(f"Current model: {st.session_state.x2img}")
        st.session_state.x2img.app()


def run_app():
    utils.create_base_page()
    x2img_app()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Args: {args}")
    logger.info(st.session_state)
    st.session_state.device = args.device
    st.session_state.output_path = args.output
    run_app()
