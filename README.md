# Image Generation and Loss Optimization with Stable Diffusion

This project generates images using the **Stable Diffusion** model and applies various loss functions to optimize image quality based on different aspects such as sharp edges, texture, entropy, symmetry, and contrast.

## Description

This script leverages **Stable Diffusion** to generate images based on a given text prompt and several predefined random seeds. The script then applies custom loss functions to the generated images and saves the results along with loss values in separate directories.

The loss functions applied are:
- **Edge Loss**: Encourages sharp edges in the image.
- **Texture Loss**: Encourages texture similarity with a random pattern.
- **Entropy Loss**: Maximizes pixel diversity in the image.
- **Symmetry Loss**: Penalizes asymmetry in the image.
- **Contrast Loss**: Encourages high contrast in the image.

Generated images, their corresponding loss values, and the modified images based on these losses are saved for each seed.

## Requirements

- Python 3.x
- PyTorch
- Huggingface Diffusers
- NumPy
- Matplotlib
- Pillow (PIL)
- torchvision

You can install the required dependencies with the following:

```bash
pip install torch torchvision diffusers numpy matplotlib pillow
```
## How to Use
- Clone this repository to your local machine.

- Run the script by executing the following command:
```
python generate_images.py
```
- The script will generate images for each seed based on the provided prompt ("A futuristic city skyline at sunset") and calculate the loss for each image.

- The images and loss values will be saved in the generated_images/ directory, organized by seed.

- For each seed, a folder is created, containing:

- The original generated image (original.png)

- Modified images based on each loss function (e.g., edge_loss.png, texture_loss.png, etc.)

- Loss value files (e.g., edge_loss.txt, texture_loss.txt, etc.)

## Huggingface Space
This project has also been deployed on Huggingface Spaces, allowing you to run the model and experiment with the generated images directly from the web interface.
[https://huggingface.co/spaces/Shriti09/futuristic-city-generator](url)

Feel free to modify the prompt and experiment with different seeds and loss functions to observe the results!

## Loss Function Details
- Edge Loss: Encourages sharp edges in the image by using Sobel filters.

- Texture Loss: Compares the image to a randomly generated pattern, promoting texture diversity.

- Entropy Loss: Maximizes pixel diversity by optimizing the image's entropy.

- Symmetry Loss: Encourages symmetry by comparing the left and right halves of the image.

- Contrast Loss: Optimizes the contrast in the image by adjusting pixel intensity distribution.

## Output
- For each seed, the script generates and saves:

- Original image

- Modified images based on each of the loss functions

- Text files containing loss values for each loss function applied
