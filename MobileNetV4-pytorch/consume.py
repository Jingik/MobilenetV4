import torch
from mobilenet.mobilenetv4 import MobileNetV4

# Support ['MobileNetV4ConvSmall', 'MobileNetV4ConvMedium', 'MobileNetV4ConvLarge']
# Also supported ['MobileNetV4HybridMedium', 'MobileNetV4HybridLarge']
model = MobileNetV4("MobileNetV4ConvSmall")

# Check the trainable params
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# Check the model's output shape
print("Check output shape ...")
x = torch.rand(1, 3, 224, 224)
y = model(x)
for i in y:
    print(i.shape)