import os
import tempfile

import streamlit as st

from groq import APIError, RateLimitError
from PIL import UnidentifiedImageError

from shopping_agent import agent, describe_product_image


def get_response(messages):
    """Show API failures without discarding chat history or retrying an order."""
    try:
        result = agent.invoke({"messages": [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]})
        return result["messages"][-1].text.replace("`", "")
    except RateLimitError:
        st.warning("Groq’s rate limit was reached. Wait a minute, then try again. "
                   "If this continues, lower SHOPPING_MAX_TOKENS in .env and restart Streamlit.")
        st.info("Review Orders in store.db before repeating a checkout request.")
        st.stop()
    except APIError as exc:
        st.error(f"Groq request failed: {exc}")
        st.info("Check your API key, model access, and rate limits. Review Orders in store.db before repeating a checkout request.")
        st.stop()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛒", layout="wide")

st.title("🛒 AI Shopping Assistant")
st.caption("Tell me what you want — I'll search, rate, and order the best match for you.")

# ---------------------------------------------------------------------------
# Sidebar — shop by image
# ---------------------------------------------------------------------------
# Initialize state before sidebar actions can append messages.
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Shop by Image")
    st.caption("Upload a photo of a product and I'll find similar items in our store.")

    uploaded_file = st.file_uploader(
        "Upload product image", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

    if uploaded_file and st.button("Find similar products", use_container_width=True):
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        prompt = f"I uploaded a product image. Please analyze it and find similar products in the store. Image path: {image_path}"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_image = uploaded_file.name
        st.rerun()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------

# Render history — show a friendlier label for image-search messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and (msg.get("image_name") or msg["content"].startswith("I uploaded a product image")):
            filename = msg.get("image_name", msg["content"].split("Image path:")[-1].strip())
            st.markdown(f"Searching by image: **{os.path.basename(filename)}**")
        else:
            st.markdown(msg["content"].replace("$", r"\$"))

# ---------------------------------------------------------------------------
# Run agent if there's an unprocessed message (image upload triggers this)
# ---------------------------------------------------------------------------
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and "pending_image" in st.session_state
):
    # Consume the trigger before calling the agent so reruns cannot repeat it.
    image_path = st.session_state.pop("pending_image")
    with st.chat_message("assistant"):
        with st.spinner("Analyzing image and searching…"):
            response = get_response(st.session_state.messages)
        st.markdown(response.replace("$", r"\$"))

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ---------------------------------------------------------------------------
# Text input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("e.g. I want organic honey under $15 with 4+ rating"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = get_response(st.session_state.messages)
        st.markdown(response.replace("$", r"\$"))

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
