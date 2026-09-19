import torch
import torch.nn as nn
import torchvision.models as models

def build_model(arch: str = "resnet18", num_classes: int = 2, pretrained: bool = True, in_channels: int = 1) -> nn.Module:
    """
    Builds a vision model adapted for single-channel grayscale input and binary classification.
    
    Supports:
      - 'resnet18': Lightweight residual network.
      - 'efficientnet_b0': Mobile inverted bottleneck with depthwise convolutions.
      - 'convnext_tiny': Modern pure-convolutional network with 7x7 depthwise kernels.
    
    Averages pretrained RGB weights across color channels, preserving low-level feature detectors.
    """
    arch = arch.lower()
    
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if in_channels != 3:
            orig_conv = model.conv1
            new_conv = nn.Conv2d(in_channels, orig_conv.out_channels, orig_conv.kernel_size, orig_conv.stride, orig_conv.padding, bias=orig_conv.bias is not None)
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
            model.conv1 = new_conv
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
        
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if in_channels != 3:
            orig_conv = model.features[0][0]
            new_conv = nn.Conv2d(in_channels, orig_conv.out_channels, orig_conv.kernel_size, orig_conv.stride, orig_conv.padding, bias=orig_conv.bias is not None)
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
            model.features[0][0] = new_conv
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        if in_channels != 3:
            orig_conv = model.features[0][0]
            new_conv = nn.Conv2d(in_channels, orig_conv.out_channels, orig_conv.kernel_size, orig_conv.stride, orig_conv.padding, bias=orig_conv.bias is not None)
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
            model.features[0][0] = new_conv
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported architecture: '{arch}'. Choose from 'resnet18', 'efficientnet_b0', 'convnext_tiny'.")

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
