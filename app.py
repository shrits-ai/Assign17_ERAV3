import os
import torch
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from diffusers import StableDiffusionPipeline

# Define Loss Functions (same as in your code)
def edge_loss(image_tensor):
    grayscale = image_tensor.mean(dim=0, keepdim=True)
    grayscale = grayscale.unsqueeze(0)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=image_tensor.device).float().unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_x.transpose(2, 3)
    gx = F.conv2d(grayscale, sobel_x, padding=1)
    gy = F.conv2d(grayscale, sobel_y, padding=1)
    return -torch.mean(torch.sqrt(gx ** 2 + gy ** 2))

def texture_loss(image_tensor):
    return F.mse_loss(image_tensor, torch.rand_like(image_tensor, device=image_tensor.device))

def entropy_loss(image_tensor):
    hist = torch.histc(image_tensor, bins=256, min=0, max=255)
    hist = hist / hist.sum()
    return -torch.sum(hist * torch.log(hist + 1e-7))

def symmetry_loss(image_tensor):
    width = image_tensor.shape[-1]
    left_half = image_tensor[:, :, :width // 2]
    right_half = torch.flip(image_tensor[:, :, width // 2:], dims=[-1])
    return F.mse_loss(left_half, right_half)

def contrast_loss(image_tensor):
    min_val = image_tensor.min()
    max_val = image_tensor.max()
    return -torch.mean((image_tensor - min_val) / (max_val - min_val + 1e-7))

# Setup Stable Diffusion Pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

# Image transform to tensor
transform = transforms.ToTensor()

# Loss functions dictionary
losses = {
    "edge": edge_loss,
    "texture": texture_loss,
    "entropy": entropy_loss,
    "symmetry": symmetry_loss,
    "contrast": contrast_loss
}

# Define function to generate images for a given seed
def generate_images(seed):
    generator = torch.Generator(device).manual_seed(seed)
    output_image = pipe("A futuristic city skyline at sunset", generator=generator).images[0]

    # Convert to tensor
    image_tensor = transform(output_image).to(device)

    loss_images = []
    loss_values = []

    # Compute losses and generate modified images
    for loss_name, loss_fn in losses.items():
        loss_value = loss_fn(image_tensor)

        # Resize to thumbnail size
        thumbnail_image = output_image.copy()
        thumbnail_image.thumbnail((128, 128))

        # Save loss image with thumbnail
        loss_images.append(thumbnail_image)
        loss_values.append(f"{loss_name}: {loss_value.item():.4f}")

    return loss_images, loss_values

# Gradio Interface
def gradio_interface(seed):
    loss_images, loss_values = generate_images(int(seed))
    return loss_images, loss_values

# Set up Gradio UI
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.inputs.Textbox(label="Enter Seed"),
    outputs=[gr.outputs.Gallery(label="Loss Images"), gr.outputs.Textbox(label="Loss Values")]
)

# Launch the interface
interface.launch()
