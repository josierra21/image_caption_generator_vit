# ViT-GPT2 Image Caption Generator

This project uses a pretrained ViT-GPT2 image captioning model in a Streamlit app.

The model comes from Hugging Face:

https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

## How It Works

```text
Image upload
ViT image encoder
GPT-2 text decoder
Generated caption
```

Unlike the earlier CNN + LSTM projects, this app does not train a model from scratch. It loads a pretrained vision-language model and uses it directly for caption generation.

## Project Structure

```text
image_caption_generator_vit/
  app.py
  requirements.txt
  runtime.txt
  README.md
```

## Setup

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the App

You can use the deployed app here:

https://joannas-image-caption-generator3.streamlit.app/

```bash
streamlit run app.py
```

The first run may take a little longer because the pretrained model is downloaded.
