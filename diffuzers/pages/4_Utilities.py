import streamlit as st

from diffuzers.image_info import ImageInfo


def app():
    task = st.selectbox(
        "Choose a utility",
        [
            "ImageInfo",
        ],
    )
    # add a line break
    st.markdown("<hr>", unsafe_allow_html=True)
    if task == "ImageInfo":
        ImageInfo().app()


if __name__ == "__main__":
    app()
