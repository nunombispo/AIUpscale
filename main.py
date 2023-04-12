import torch
from PIL import Image
from RealESRGAN import RealESRGAN

# Device CPU
device = torch.device('cpu')

# Load model and scale factor
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

# Load image
path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

# Upscale image
sr_image = model.predict(image)

# Save image
sr_image.save('results/sr_image.png')

