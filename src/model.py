"""
model.py

U-Net with ResNet-50 encoder (ImageNet pretrained).

Architecture:
  Encoder: ResNet-50 layers 0-4 (pretrained, fine-tuned end-to-end)
  Bottleneck: layer4 output (2048 channels)
  Decoder: 4 upsampling blocks with skip connections
  Head: 1x1 conv → NUM_CLASSES channels → softmax (at inference)

Skip connections from encoder stages 0-3 are concatenated
with the corresponding decoder stage outputs.

Also includes MC Dropout variant for uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import NUM_CLASSES


# ── Decoder building block ────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Upsample by 2x, concatenate skip connection, then two conv-BN-ReLU layers.
    in_channels  : channels coming from the layer below (upsampled)
    skip_channels: channels from the encoder skip connection
    out_channels : output channels
    dropout_p    : dropout probability (0 = disabled, >0 = MC Dropout enabled)
    """

    def __init__(self, in_channels, skip_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                     align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch (odd input dimensions)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── U-Net ─────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net with ResNet-50 encoder.

    Args:
        num_classes  : number of output classes (default 6 for sweet pepper)
        pretrained   : use ImageNet pretrained ResNet-50 encoder
        dropout_p    : dropout probability in decoder blocks
                       Set > 0 to enable MC Dropout at inference time.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout_p=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p   = dropout_p

        # ── Encoder (ResNet-50) ───────────────────────────────────────────────
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Stage 0: conv1 + bn1 + relu (stride 2) → (64, H/2, W/2)
        self.enc0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )
        # Stage 1: maxpool + layer1 (stride 2) → (256, H/4, W/4)
        self.enc1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        # Stage 2: layer2 (stride 2) → (512, H/8, W/8)
        self.enc2 = backbone.layer2
        # Stage 3: layer3 (stride 2) → (1024, H/16, W/16)
        self.enc3 = backbone.layer3
        # Bottleneck: layer4 (stride 2) → (2048, H/32, W/32)
        self.bottleneck = backbone.layer4

        # ── Decoder ───────────────────────────────────────────────────────────
        # Each block: upsample + concat skip + conv
        # Channels: (from_below, skip_channels, out)
        self.dec4 = DecoderBlock(2048, 1024, 512,  dropout_p)
        self.dec3 = DecoderBlock(512,  512,  256,  dropout_p)
        self.dec2 = DecoderBlock(256,  256,  128,  dropout_p)
        self.dec1 = DecoderBlock(128,  64,   64,   dropout_p)

        # Final upsample back to input resolution + classification head
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                           align_corners=True)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)         # (B, 64,   H/2,  W/2)
        e1 = self.enc1(e0)        # (B, 256,  H/4,  W/4)
        e2 = self.enc2(e1)        # (B, 512,  H/8,  W/8)
        e3 = self.enc3(e2)        # (B, 1024, H/16, W/16)
        b  = self.bottleneck(e3)  # (B, 2048, H/32, W/32)

        # Decoder with skip connections
        d4 = self.dec4(b,  e3)   # (B, 512,  H/16, W/16)
        d3 = self.dec3(d4, e2)   # (B, 256,  H/8,  W/8)
        d2 = self.dec2(d3, e1)   # (B, 128,  H/4,  W/4)
        d1 = self.dec1(d2, e0)   # (B, 64,   H/2,  W/2)

        # Final upsample to input resolution
        out = self.final_upsample(d1)   # (B, 64,  H, W)
        out = self.head(out)            # (B, num_classes, H, W)
        return out                      # raw logits

    def predict_proba(self, x):
        """Returns softmax probabilities (B, num_classes, H, W)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict_mask(self, x):
        """Returns argmax class predictions (B, H, W)."""
        return self.predict_proba(x).argmax(dim=1)

    def predict_entropy(self, x):
        """
        Returns per-pixel predictive entropy (B, H, W).
        High entropy = uncertain. Low entropy = confident.
        """
        proba   = self.predict_proba(x)
        entropy = -(proba * torch.clamp(proba, min=1e-7).log()).sum(dim=1)
        return entropy

    def enable_dropout(self):
        """
        Keeps Dropout2d layers in train mode even when model.eval() is called.
        Call this before running MC Dropout inference.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()

    def count_parameters(self):
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(pretrained=True, dropout_p=0.0, device="cuda"):
    """Convenience function — build and move to device."""
    model = UNet(num_classes=NUM_CLASSES,
                 pretrained=pretrained,
                 dropout_p=dropout_p)
    model = model.to(device)
    total, trainable = model.count_parameters()
    print(f"Model: U-Net ResNet-50 | "
          f"params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    return model


if __name__ == "__main__":
    # Quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(pretrained=False, dropout_p=0.3, device=device)
    x      = torch.randn(2, 3, 512, 512).to(device)

    logits  = model(x)
    proba   = model.predict_proba(x)
    mask    = model.predict_mask(x)
    entropy = model.predict_entropy(x)

    print(f"Input:   {x.shape}")
    print(f"Logits:  {logits.shape}")
    print(f"Proba:   {proba.shape}  (sums to {proba.sum(1).mean():.3f} per pixel)")
    print(f"Mask:    {mask.shape}   (classes 0-{NUM_CLASSES-1})")
    print(f"Entropy: {entropy.shape} (mean={entropy.mean():.3f})")
    print("Model OK")