import streamlit as st

from diffuzers.textual_inversion import TextualInversion


def app():
    with st.form("textual_inversion_form"):
        model = st.text_input("Which base model do you want to use?", value="CompVis/stable-diffusion-v1-4")
        token_identifier = st.text_input("Token identifier", value="<something>")
        embeddings = st.text_input(
            "Embeddings", value="https://huggingface.co/sd-concepts-library/axe-tattoo/resolve/main/learned_embeds.bin"
        )
        # st.file_uploader("Embeddings", type=["pt", "bin"])
        submit = st.form_submit_button("Load model")
    if submit:
        with st.spinner("Loading model..."):
            textual_inversion = TextualInversion(
                model=model,
                token_identifier=token_identifier,
                embeddings_url=embeddings,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.textual_inversion = textual_inversion
    if "textual_inversion" in st.session_state:
        st.write(f"Current model: {st.session_state.textual_inversion}")
        st.session_state.textual_inversion.app()


if __name__ == "__main__":
    app()
