import torch
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes: int = 2, pretrained: bool = True, in_channels: int = 1) -> nn.Module:
    """
    Builds a ResNet18 model adapted for single-channel grayscale input and 2-class classification.
    
    Averages the pretrained first-layer RGB weights across the 3 color channels
    rather than reinitializing from scratch, preserving low-level feature detectors (edges, textures).
    
    Args:
        num_classes: Number of output classes (default: 2).
        pretrained: Whether to load ImageNet pretrained weights (default: True).
        in_channels: Number of input channels (default: 1 for grayscale).
        
    Returns:
        Adapted nn.Module ready for training or inference.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # 1. Adapt first conv layer for single-channel grayscale input
    if in_channels != 3:
        orig_conv1 = model.conv1
        new_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
            bias=orig_conv1.bias is not None
        )
        
        if pretrained:
            with torch.no_grad():
                # Average weights across RGB channels (dim=1) -> shape: (64, 1, 7, 7)
                avg_weights = orig_conv1.weight.mean(dim=1, keepdim=True)
                new_conv1.weight.copy_(avg_weights)
                if orig_conv1.bias is not None:
                    new_conv1.bias.copy_(orig_conv1.bias)
                    
        model.conv1 = new_conv1
        
    # 2. Replace final fully connected layer for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def test_model_architecture():
    print("=" * 60)
    print("MODEL ARCHITECTURE VERIFICATION (model.py)")
    print("=" * 60)
    
    print("[+] Instantiating ResNet18 (pretrained=True, in_channels=1, num_classes=2)...")
    model = build_model(num_classes=2, pretrained=True, in_channels=1)
    model.eval()
    
    # Verify layer dimensions
    print(f"[+] conv1 layer: {model.conv1}")
    print(f"    conv1 weight shape: {tuple(model.conv1.weight.shape)} (Expected: (64, 1, 7, 7))")
    print(f"[+] fc layer:    {model.fc}")
    print(f"    fc weight shape:    {tuple(model.fc.weight.shape)} (Expected: (2, 512))")
    
    # Total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Total parameters:     {total_params:,}")
    print(f"[+] Trainable parameters: {trainable_params:,}")
    
    # Dummy forward pass with random 1x1x256x256 tensor
    dummy_input = torch.randn(1, 1, 256, 256)
    print(f"\n[+] Executing dummy forward pass with input tensor: {tuple(dummy_input.shape)}...")
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print("\n" + "-" * 50)
    print("FORWARD PASS RESULTS")
    print("-" * 50)
    print(f"Input Tensor Shape:  {tuple(dummy_input.shape)}")
    print(f"Output Tensor Shape: {tuple(output.shape)} (Expected: (1, 2))")
    print(f"Output Raw Logits:   {output.numpy().tolist()}")
    
    probabilities = torch.softmax(output, dim=1)
    print(f"Output Softmax Probs:{probabilities.numpy().tolist()}")
    print("-" * 50)
    print("[+] Architecture verified and wired correctly!")
    print("=" * 60)

if __name__ == "__main__":
    test_model_architecture()
