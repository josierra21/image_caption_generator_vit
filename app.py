import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

st.set_page_config(page_title="ViT-GPT2 Image Caption Generator", layout="centered")
st.title("ViT-GPT2 Image Caption Generator")
st.write(
    "Upload a photo and the app will generate a caption using a pretrained ViT-GPT2 model. "
    "Unlike the earlier CNN + LSTM projects, this model uses a vision transformer to read the image "
    "and GPT-2 to write a more natural caption."
)


@st.cache_resource
def load_caption_model():
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor, tokenizer, device


def generate_caption(image: Image.Image, model, processor, tokenizer, device: str) -> str:
    image = image.convert("RGB")
    pixel_values = processor(images=[image], return_tensors="pt").pixel_values.to(device)

    output_ids = model.generate(
        pixel_values,
        max_length=24,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


model, processor, tokenizer, device = load_caption_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

    if st.button("Generate caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(uploaded_image, model, processor, tokenizer, device)
        st.subheader("Generated caption")
        st.write(caption if caption else "No caption generated. Try another image.")
else:
    st.info("Upload a JPG or PNG image to generate a caption.")
