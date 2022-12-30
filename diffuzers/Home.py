import argparse

import streamlit as st
from loguru import logger


CODE_OF_CONDUCT = """
## Code of conduct
The app should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

Using the app to generate content that is cruel to individuals is a misuse of this app. One shall not use this app to generate content that is intended to be cruel to individuals, or to generate content that is intended to be cruel to individuals in a way that is not obvious to the viewer.
This includes, but is not limited to:
- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.

By using this app, you agree to the above code of conduct.

"""


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


def run_app():
    st.title("Diffuzers")
    st.markdown("Welcome to Diffuzers! A web app for [🤗 Diffusers](https://github.com/huggingface/diffusers)")
    st.markdown("")
    st.markdown(CODE_OF_CONDUCT)


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Args: {args}")
    logger.info(st.session_state)
    st.session_state.device = args.device
    st.session_state.output_path = args.output
    # text2img, img2img, inpainting = get_models(args)
    run_app()
