import gradio as gr
import torch
from diffusers import FluxPipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the model
print("Loading FLUX model with LoRA...")

# Get Hugging Face token from environment
hf_token = os.getenv("HF_TOKEN")

# Load the base FLUX.1-Kontext-dev model
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    token=hf_token
)

# Load the YouTube Thumbnails LoRA
pipe.load_lora_weights(
    "fal/Youtube-Thumbnails-Kontext-Dev-LoRA",
    adapter_name="youtube_thumbnails"
)

# Enable model optimizations
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    pipe = pipe.to("mps")
    print("Using MPS (Apple Silicon)")
else:
    print("Using CPU (this will be slow)")

# Set LoRA scale
pipe.set_adapters(["youtube_thumbnails"], adapter_weights=[0.45])

print("Model loaded successfully!")

def generate_thumbnail(thumbnail_text, additional_description="", lora_scale=0.45, num_inference_steps=30, guidance_scale=3.5):
    """
    Generate a YouTube thumbnail with custom text

    Args:
        thumbnail_text: The text to appear on the thumbnail (e.g., "EPIC FAIL")
        additional_description: Optional additional prompt elements (e.g., "with hand pointing")
        lora_scale: LoRA weight (0.4-0.5 recommended)
        num_inference_steps: Number of denoising steps (higher = better quality but slower)
        guidance_scale: How closely to follow the prompt
    """
    # Construct the prompt using the recommended format
    prompt = f"Generate youtube thumbnails using text '{thumbnail_text}'"
    if additional_description:
        prompt += f", {additional_description}"

    # Update LoRA scale if changed
    pipe.set_adapters(["youtube_thumbnails"], adapter_weights=[lora_scale])

    # Generate the image
    print(f"Generating with prompt: {prompt}")
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=1536,
        height=1024,
        generator=torch.Generator().manual_seed(42)
    )

    return result.images[0]

# Create Gradio interface
with gr.Blocks(title="YouTube Thumbnail Generator") as demo:
    gr.Markdown("""
    # YouTube Thumbnail Generator
    Generate eye-catching YouTube thumbnails using FLUX.1-Kontext-dev with custom text overlay!

    **Recommended settings:**
    - LoRA Scale: 0.4-0.5
    - Resolution: 1536x1024 (YouTube standard)
    """)

    with gr.Row():
        with gr.Column():
            thumbnail_text = gr.Textbox(
                label="Thumbnail Text",
                placeholder="e.g., EPIC FAIL, MIND BLOWN, CAN'T BELIEVE IT!",
                value="EPIC FAIL"
            )
            additional_desc = gr.Textbox(
                label="Additional Description (Optional)",
                placeholder="e.g., with hand pointing, shocked expression",
                value=""
            )

            with gr.Accordion("Advanced Settings", open=False):
                lora_scale = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.45,
                    step=0.05,
                    label="LoRA Scale"
                )
                num_steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=30,
                    step=5,
                    label="Inference Steps"
                )
                guidance = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.5,
                    step=0.5,
                    label="Guidance Scale"
                )

            generate_btn = gr.Button("Generate Thumbnail", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Thumbnail", type="pil")

    # Example prompts
    gr.Examples(
        examples=[
            ["MIND BLOWN", "with explosion effects"],
            ["EPIC FAIL", "shocked expression"],
            ["CAN'T BELIEVE IT!", "hand pointing"],
            ["SHOCKING NEWS", "dramatic lighting"],
            ["YOU WON'T BELIEVE THIS", "surprised face"],
        ],
        inputs=[thumbnail_text, additional_desc],
    )

    generate_btn.click(
        fn=generate_thumbnail,
        inputs=[thumbnail_text, additional_desc, lora_scale, num_steps, guidance],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch()

