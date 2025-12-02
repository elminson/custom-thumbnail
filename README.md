---
title: Custom Thumbnail
emoji: ⚡
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 4.25.0
app_file: app.py
pinned: false
thumbnail: https://img.favpng.com/0/14/12/llama-png-favpng-bWH9wH8CN20SkvkH1cVYRFN1V.jpg

---

# YouTube Thumbnail Generator

Generate eye-catching YouTube thumbnails using FLUX.1-Kontext-dev with the YouTube Thumbnails LoRA adapter.

This application uses the [fal/Youtube-Thumbnails-Kontext-Dev-LoRA](https://huggingface.co/fal/Youtube-Thumbnails-Kontext-Dev-LoRA) model to create professional-looking thumbnails with custom text overlays.

## Features

- Generate 1536x1024 thumbnails (YouTube standard resolution)
- Custom text overlay with various styles
- Adjustable LoRA scale for fine-tuning (recommended: 0.4-0.5)
- Advanced settings for inference steps and guidance scale
- Pre-built example prompts to get started quickly

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended) or Apple Silicon with MPS support
- Hugging Face account with access token

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd custom_thumbnail
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Hugging Face token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with read access
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your token:
     ```
     HF_TOKEN=your_actual_token_here
     ```

5. Accept the FLUX.1-Kontext-dev license:
   - Visit https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
   - Click "Agree and access repository"

## Usage

Run the application:
```bash
python app.py
```

The Gradio interface will open in your browser (usually at http://127.0.0.1:7860).

### How to Use:

1. **Thumbnail Text**: Enter the text you want on your thumbnail (e.g., "EPIC FAIL", "MIND BLOWN")
2. **Additional Description** (optional): Add extra details like "with hand pointing" or "shocked expression"
3. **Advanced Settings**: Adjust LoRA scale, inference steps, and guidance scale for fine-tuning
4. Click **Generate Thumbnail**

### Example Prompts:

- "MIND BLOWN" with "explosion effects"
- "EPIC FAIL" with "shocked expression"
- "CAN'T BELIEVE IT!" with "hand pointing"
- "SHOCKING NEWS" with "dramatic lighting"

## System Requirements

- **GPU Memory**: At least 8GB VRAM recommended
- **RAM**: 16GB+ recommended
- **Storage**: ~20GB for model weights

### Performance Notes:

- **CUDA GPU**: Fast generation (~30-60 seconds)
- **Apple Silicon (MPS)**: Moderate speed
- **CPU**: Very slow (not recommended for production use)

## Model Information

- **Base Model**: [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- **LoRA Adapter**: [fal/Youtube-Thumbnails-Kontext-Dev-LoRA](https://huggingface.co/fal/Youtube-Thumbnails-Kontext-Dev-LoRA)
- **License**: flux1-dev-non-commercial-license
- **Optimal Resolution**: 1536x1024 pixels

## Troubleshooting

### "Out of memory" errors:
- Reduce the inference steps
- Close other GPU-intensive applications
- Use a smaller batch size or lower resolution (not recommended for thumbnails)

### Slow generation:
- Ensure you're using GPU acceleration (CUDA or MPS)
- Reduce the number of inference steps (try 20-25)

### Model download issues:
- Verify your HF_TOKEN is set correctly in `.env`
- Ensure you've accepted the FLUX.1-Kontext-dev license
- Check your internet connection

## License

This project uses the FLUX.1-dev model which is licensed under the flux1-dev-non-commercial-license. See the [model card](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) for details.

## Acknowledgments

- Model trained by [fal.ai](https://fal.ai/)
- Base model by [Black Forest Labs](https://blackforestlabs.ai/)
- Built with [Gradio](https://gradio.app/) and [Diffusers](https://huggingface.co/docs/diffusers/)
