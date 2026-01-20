import os
import torch
import gradio as gr
from PIL import Image

from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor

MODEL_DIR = os.getenv("MODEL_DIR", ".")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(device)
image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model.eval()

def caption_image(
    img: Image.Image,
    max_new_tokens=30,
    num_beams=4,
    do_sample=False,
    temperature=1.0,
    top_p=0.9,
):
    if img is None:
        return "No image provided."

    img = img.convert("RGB")
    pixel_values = image_processor(images=img, return_tensors="pt").pixel_values.to(device)

    gen_kwargs = {"max_new_tokens": int(max_new_tokens)}

    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_beams": 1,   
        })
    else:
        gen_kwargs.update({
            "do_sample": False,
            "num_beams": int(num_beams),
        })

    with torch.no_grad():
        output_ids = model.generate(pixel_values, **gen_kwargs)

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption if caption else "(empty caption)"

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="zinc",
).set(
    body_background_fill="#0f172a",      # dark blue
    block_background_fill="#020617",     # card background
    block_border_color="#1e293b",
    input_background_fill="#020617",
    button_primary_background_fill="#6366f1",
    button_primary_background_fill_hover="#4f46e5",
    button_primary_text_color="white",
)


with gr.Blocks(theme=theme) as demo:

    gr.Markdown(
        """
        <h1 style='text-align:center;'>Image Captioning</h1>
        <p style='text-align:center; color: #94a3b8;'>
        Understanding Images Through Language.
        </p>
        """,
        elem_id="title"
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            inp = gr.Image(
                type="pil",
                label="Upload Image",
                height=350
            )

        with gr.Column(scale=1):
            out = gr.Textbox(
                label="Generated Caption",
                lines=6,
                placeholder="The caption will appear here..."
            )
            btn = gr.Button("Generate Caption", size="lg")

    with gr.Accordion("Generation Settings", open=False):
        with gr.Row():
            max_new_tokens = gr.Slider(5, 80, value=30, step=1, label="Max New Tokens")
            num_beams = gr.Slider(1, 10, value=4, step=1, label="Beam Search")

        with gr.Row():
            do_sample = gr.Checkbox(value=False, label="Enable Sampling (Creative Mode)")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")

    btn.click(
        fn=caption_image,
        inputs=[inp, max_new_tokens, num_beams, do_sample, temperature, top_p],
        outputs=out,
    )

    gr.Markdown(
        """
        <p style='text-align:center; color:#64748b; font-size: 12px;'>
        ViT Encoder • GPT-2 Decoder • Transformers
        </p>
        """
    )


if __name__ == "__main__":
    demo.launch()
