import torch.nn as nn
import torch

class LinearFusion(nn.Module):
    def __init__(self, in_dim=768):
        super().__init__()
        self.attention = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: [200, 5, 1024]
        weights = torch.softmax(self.attention(x).squeeze(2), dim=1)
        weights = weights.unsqueeze(2)
        x_fused = (x * weights).sum(dim=1)
        return x_fused


class Evidence(nn.Module):
    def __init__(self, downsample_dim=1024):
        super().__init__()

        self.image_uncertainty_head = nn.Sequential(
            nn.Linear(downsample_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        self.image_proj = nn.Linear(downsample_dim, downsample_dim)
        self.fusion_img = LinearFusion(in_dim=downsample_dim)

    def forward(self, img_features):
        image_logits = self.image_uncertainty_head(img_features)
        image_evidence = torch.exp(image_logits)
        image_alpha = image_evidence + 1
        K = image_alpha.shape[-1]
        image_S = image_alpha.sum(dim=-1)
        image_u = K / image_S
        image_weights = 1.0 - image_u
        image_weighted = (img_features * image_weights.unsqueeze(-1)).sum(dim=1)
        image_weights_sum = image_weights.sum(dim=1, keepdim=True) + 1e-8
        image_fused = image_weighted / image_weights_sum
        image_proj = self.image_proj(image_fused)
        final_features = image_proj + self.fusion_img(img_features)
        return final_features


