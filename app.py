import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr
from diffusers import StableDiffusionPipeline

# Define Loss Functions (same as your original code)

# Setup Stable Diffusion Pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

# Image transform to tensor
transform = transforms.ToTensor()

# Create a function to generate images
def generate_images(prompt, seed=42):
    # Set generator with a fixed seed
    generator = torch.Generator(device).manual_seed(seed)
    
    # Generate image
    output_image = pipe(prompt, generator=generator).images[0]
    
    # Convert to tensor
    image_tensor = transform(output_image).to(device)

    # Compute losses (similar to original code)
    losses = {
        "edge": edge_loss,
        "texture": texture_loss,
        "entropy": entropy_loss,
        "symmetry": symmetry_loss,
        "contrast": contrast_loss
    }

    # Save and return the image
    result_image_path = f"generated_images/seed_{seed}/original.png"
    output_image.save(result_image_path)
    
    # Return the image for display
    return output_image

# Define Gradio Interface
interface = gr.Interface(
    fn=generate_images,  # Function to call when user interacts with the UI
    inputs=[
        gr.Textbox(label="Prompt", value="A futuristic city skyline at sunset"),
        gr.Slider(minimum=0, maximum=1000, step=1, label="Seed", value=42)
    ],
    outputs=gr.Image(type="pil"),
    live=True,
    title="Futuristic City Image Generator",
    description="Generate images of futuristic cities with various customizable settings."
)

# Launch the interface
interface.launch()
