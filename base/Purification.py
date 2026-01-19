import torch.nn as nn
import torch


class Purification(nn.Module):

    def __init__(self, input_dim, output_dim, bottleneck_ratio=0.25, dropout_rate=0.3):

        super().__init__()

        self.input_dim = input_dim
        
        bottleneck_dim = int(input_dim * bottleneck_ratio)
        
        self.adapter = nn.Sequential(

            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),  
            nn.Linear(bottleneck_dim, output_dim),
            nn.Dropout(dropout_rate)  
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.shape[-1] == self.input_dim, "Input dimension mismatch"

        adapter_output = self.adapter(x)
        final_output = self.layer_norm(adapter_output)
        
        return final_output


