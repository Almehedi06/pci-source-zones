from __future__ import annotations

from typing import Any


def build_unet(model_cfg: dict[str, Any], in_channels: int, regression: bool = False):
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Install torch to use UNet: pip install torch") from exc

    return UNet(
        in_channels=in_channels,
        base_filters=int(model_cfg.get("base_filters", 32)),
        depth=int(model_cfg.get("depth", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        regression=regression,
        attention=bool(model_cfg.get("attention", False)),
        tobit=bool(model_cfg.get("tobit", False)),
    )


def _make_conv_block(in_ch: int, out_ch: int, dropout: float = 0.0):
    import torch.nn as nn
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(p=dropout))
    return nn.Sequential(*layers)


class UNet:
    """Placeholder — actual class defined below when torch is available."""


def _build_unet_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class AttentionGate(nn.Module):
        """Soft attention on skip connections (Oktay et al. 2018).

        Learns which spatial locations in the skip feature are relevant
        given the decoder gating signal — suppresses flat ridges, focuses
        on convergence zones and steep slopes.
        """
        def __init__(self, x_ch: int, g_ch: int) -> None:
            super().__init__()
            inter_ch = max(x_ch // 2, 1)
            self.W_x   = nn.Conv2d(x_ch,    inter_ch, kernel_size=1)
            self.W_g   = nn.Conv2d(g_ch,    inter_ch, kernel_size=1)
            self.psi   = nn.Conv2d(inter_ch, 1,        kernel_size=1)
            self.bn    = nn.BatchNorm2d(1)
            self.relu  = nn.ReLU(inplace=True)
            self.sigma = nn.Sigmoid()

        def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            # x = skip connection, g = upsampled decoder (gating signal)
            # both have same spatial size after upconv
            alpha = self.sigma(self.bn(self.psi(self.relu(self.W_x(x) + self.W_g(g)))))
            return x * alpha

    class UNet(nn.Module):
        """Encoder-decoder UNet with optional attention gates on skip connections.

        in_channels : number of input feature channels (one per raster band)
        base_filters : filters in first encoder block; doubles each level
        depth        : number of encoder/decoder levels
        dropout      : spatial dropout applied inside each conv block
        attention    : if True, adds soft attention gates on all skip connections
        """

        def __init__(
            self,
            in_channels: int,
            base_filters: int = 32,
            depth: int = 4,
            dropout: float = 0.0,
            regression: bool = False,
            attention: bool = False,
            tobit: bool = False,
        ) -> None:
            super().__init__()
            self.depth     = depth
            self.regression = regression
            self.use_attention = attention
            self.tobit = tobit

            # Encoder
            self.encoders = nn.ModuleList()
            self.pools    = nn.ModuleList()
            ch = in_channels
            self._enc_channels: list[int] = []
            for i in range(depth):
                out_ch = base_filters * (2 ** i)
                self.encoders.append(_make_conv_block(ch, out_ch, dropout))
                self.pools.append(nn.MaxPool2d(2))
                self._enc_channels.append(out_ch)
                ch = out_ch

            # Bottleneck
            btn_ch = base_filters * (2 ** depth)
            self.bottleneck = _make_conv_block(ch, btn_ch, dropout)
            ch = btn_ch

            # Decoder
            self.upconvs   = nn.ModuleList()
            self.decoders  = nn.ModuleList()
            self.att_gates = nn.ModuleList()
            for i in range(depth - 1, -1, -1):
                skip_ch = self._enc_channels[i]
                self.upconvs.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
                if attention:
                    # gating signal (g) has skip_ch channels after upconv
                    self.att_gates.append(AttentionGate(x_ch=skip_ch, g_ch=skip_ch))
                self.decoders.append(_make_conv_block(skip_ch * 2, skip_ch, dropout))
                ch = skip_ch

            out_ch = 2 if tobit else 1
            self.head = nn.Conv2d(ch, out_ch, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            skips: list[torch.Tensor] = []
            for enc, pool in zip(self.encoders, self.pools):
                x = enc(x)
                skips.append(x)
                x = pool(x)

            x = self.bottleneck(x)

            att_iter = iter(self.att_gates) if self.use_attention else None
            for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
                x = upconv(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                if att_iter is not None:
                    skip = next(att_iter)(skip, x)
                x = torch.cat([x, skip], dim=1)
                x = dec(x)

            out = self.head(x)
            return out if self.regression else torch.sigmoid(out)

    return UNet


try:
    import torch  # noqa: F401
    UNet = _build_unet_class()
except ImportError:
    pass
