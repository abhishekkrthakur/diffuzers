import streamlit as st

from diffuzers import utils


def app():
    utils.create_base_page()
    st.markdown(utils.CODE_OF_CONDUCT)


if __name__ == "__main__":
    app()
