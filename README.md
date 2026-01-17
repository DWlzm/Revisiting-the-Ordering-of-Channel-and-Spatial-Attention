# Revisiting the Ordering of Channel and Spatial Attention: A Comprehensive Study on Sequential and Parallel Designs

This paper can be accessed for free at https://arxiv.org/pdf/2601.07310.

The remaining code will be open-sourced in phases.
- Preliminaries ☑
- Sequential Mode ☑
- Parallel Mode ☑
- Residual Connection Pattern
- Multi-scale Information Pattern
  
## Abstract

Attention mechanisms have become a core component of deep learning models, with Channel Attention and Spatial Attention being the two most representative architectures. Current research on their fusion strategies primarily bifurcates into sequential and parallel paradigms, yet the selection process remains largely empirical, lacking systematic analysis and unified principles. We systematically compare channel-spatial attention combinations under a unified framework, building an evaluation suite of 18 topologies across four classes: sequential, parallel, multi-scale, and residual. Across two vision and nine medical datasets, we uncover a "data scale-method-performance" coupling law: (1) in few-shot tasks, the "Channel-Multi-scale Spatial" cascaded structure achieves optimal performance; (2) in medium-scale tasks, parallel learnable fusion architectures demonstrate superior results; (3) in large-scale tasks, parallel structures with dynamic gating yield the best performance. Additionally, experiments indicate that the "Spatial-Channel" order is more stable and effective for fine-grained classification, while residual connections mitigate vanishing gradient problems across varying data scales. We thus propose scenario-based guidelines for building future attention modules. 

![introduction-1](https://github.com/user-attachments/assets/94d81781-4a29-48f2-b4d4-7e20b7689cc8)

## Cite

```
@misc{liu2026revisitingorderingchannelspatial,
      title={Revisiting the Ordering of Channel and Spatial Attention: A Comprehensive Study on Sequential and Parallel Designs}, 
      author={Zhongming Liu and Bingbing Jiang},
      year={2026},
      eprint={2601.07310},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.07310}, 
}
```


## Code
This section first introduces the core structures, mathematical expressions, and functional positioning of three types of basic attention components (Channel Attention, Spatial Attention, and Gate Attention)
### Preliminaries

![BasicComponents-1](https://github.com/user-attachments/assets/9824548c-a282-4124-b3ba-698661972cc6)

#### Channel Attention(CA)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel-attention module (conv-based MLP)"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = torch.sigmoid(attn)
        return x * attn

def main():
    torch.manual_seed(42)
    x = torch.randn(2, 64, 32, 32)
    ca = ChannelAttention(64, 16)
    with torch.no_grad():
        out = ca(x)
    print(f"in: {x.shape}  ->  out: {out.shape}")


if __name__ == "__main__":
    main()
```

#### Spatial Attention(SA)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Spatial-attention module"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_pool, max_pool], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn


def main():
    """Test SpatialAttention"""
    torch.manual_seed(42)
    x = torch.randn(2, 64, 32, 32)
    sa = SpatialAttention(7)
    with torch.no_grad():
        out = sa(x)
    print(f"in: {x.shape}  ->  out: {out.shape}")


if __name__ == "__main__":
    main()
```

#### Gate Attention(GA)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GateAttention(nn.Module):
    """Gate-attention module"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.mlp   = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.mlp(self.gap(x))   # (N,C,1,1)
        return x * attn                # channel-wise gate


def main():
    """Test GateAttention"""
    torch.manual_seed(42)
    x = torch.randn(2, 64, 32, 32)
    ga = GateAttention(64, 16)
    with torch.no_grad():
        out = ga(x)
    print(f"in: {x.shape}  ->  out: {out.shape}")


if __name__ == "__main__":
    main()
```

### Sequential Mode

![first-1](https://github.com/user-attachments/assets/30d25e08-00d8-44d6-99ba-6816122b2b82)


####  Channel and Spatial Attention ( CSA || CBAM )

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel-attention module (conv-based MLP)"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = torch.sigmoid(attn)
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_pool, max_pool], dim=1)
        attn = torch.sigmoid(self.conv(feat))
        return x * attn


class ChannelandSpatialAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


def main():
    torch.manual_seed(42)
    x = torch.randn(2, 64, 32, 32)
    csa = ChannelandSpatialAttention(64, 16)
    with torch.no_grad():
        out = csa(x)
    print(f"in: {x.shape}  ->  out: {out.shape}")


if __name__ == "__main__":
    main()
```

#### Spatial and Channel Attention 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel-attention module (conv-based MLP)"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = torch.sigmoid(attn)
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_pool, max_pool], dim=1)
        attn = torch.sigmoid(self.conv(feat))
        return x * attn


class SCAModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ca(self.sa(x))


def main():
    torch.manual_seed(42)
    x = torch.randn(4, 64, 32, 32)
    sca = SCAModule(64, 16)
    with torch.no_grad():
        out = sca(x)
    print(f"in: {x.shape}  ->  out: {out.shape}")


if __name__ == "__main__":
    main()
```

#### CSCA
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = torch.sigmoid(attn)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_pool, max_pool], dim=1)
        attn = torch.sigmoid(self.conv(feat))
        return x * attn


class CSCA(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.ca1 = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()
        self.ca2 = ChannelAttention(in_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ca2(self.sa(self.ca1(x)))


def main():
    torch.manual_seed(42)
    x = torch.randn(4, 64, 32, 32)
    model = CSCA(64, 16)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"in: {x.shape}  ->  out: {out.shape}")


if __name__ == "__main__":
    main()
```
#### SCSA

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = torch.sigmoid(attn)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_pool, max_pool], dim=1)
        attn = torch.sigmoid(self.conv(feat))
        return x * attn


class SCSA(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.sa1 = SpatialAttention(kernel_size=kernel_size)
        self.ca  = ChannelAttention(in_channels, reduction=reduction)
        self.sa2 = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa2(self.ca(self.sa1(x)))


def main():
    torch.manual_seed(42)
    x = torch.randn(2, 64, 32, 32)
    model = SCSA(64, 16, 7)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"in: {x.shape}  ->  out: {out.shape}")


if __name__ == "__main__":
    main()
```

### Parallel

![second-1](https://github.com/user-attachments/assets/10a206bc-174b-4ed9-8f2b-feb610e17003)


#### Channel \& Spatial Additive Attention (C\&S$A^2$)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = torch.sigmoid(attn)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_pool, max_pool], dim=1)
        attn = torch.sigmoid(self.conv(feat))
        return x * attn


class CSAParallel(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.ca(x) + self.sa(x))


def main():
    torch.manual_seed(42)
    x = torch.randn(2, 64, 32, 32)
    model = CSAParallel(64, 16)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"in: {x.shape}  ->  out: {out.shape}")


if __name__ == "__main__":
    main()
```

