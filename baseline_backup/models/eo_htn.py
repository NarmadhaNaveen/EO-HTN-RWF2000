import torch
import torch.nn as nn

class CNNStem(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)


class ContextAwareGate(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: batch, frames, features
        context = x.mean(dim=1, keepdim=True).expand_as(x)
        weights = self.gate(torch.cat([x, context], dim=-1))
        return x * weights


class EOHTN(nn.Module):
    def __init__(self, frames=8, dim=128, heads=4, layers=2):
        super().__init__()
        self.frames = frames
        self.cnn = CNNStem(dim)
        self.position = nn.Parameter(torch.zeros(1, frames, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=layers
        )
        self.context_gate = ContextAwareGate(dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, video):
        # video: batch, frames, channels, height, width
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        x = self.cnn(x)
        x = x.reshape(b, t, -1)
        x = self.transformer(x + self.position[:, :t])
        x = self.context_gate(x)
        x = x.mean(dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    model = EOHTN()
    sample = torch.randn(2, 8, 3, 96, 96)
    output = model(sample)
    print("Output shape:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
