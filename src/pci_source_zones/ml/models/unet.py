from __future__ import annotations

from typing import Any


def build_unet(model_cfg: dict[str, Any], in_channels: int):
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Install torch to use UNet: pip install torch") from exc

    return UNet(
        in_channels=in_channels,
        base_filters=int(model_cfg.get("base_filters", 32)),
        depth=int(model_cfg.get("depth", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


class _ConvBlock:
    pass


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

    class UNet(nn.Module):
        """Encoder-decoder UNet with skip connections.

        in_channels : number of input feature channels (one per raster band)
        base_filters : filters in first encoder block; doubles each level
        depth        : number of encoder/decoder levels
        dropout      : spatial dropout applied inside each conv block
        """

        def __init__(
            self,
            in_channels: int,
            base_filters: int = 32,
            depth: int = 4,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.depth = depth

            # Encoder
            self.encoders = nn.ModuleList()
            self.pools = nn.ModuleList()
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
            self.upconvs = nn.ModuleList()
            self.decoders = nn.ModuleList()
            for i in range(depth - 1, -1, -1):
                skip_ch = self._enc_channels[i]
                self.upconvs.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
                self.decoders.append(_make_conv_block(skip_ch * 2, skip_ch, dropout))
                ch = skip_ch

            self.head = nn.Conv2d(ch, 1, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            skips: list[torch.Tensor] = []
            for enc, pool in zip(self.encoders, self.pools):
                x = enc(x)
                skips.append(x)
                x = pool(x)

            x = self.bottleneck(x)

            for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
                x = upconv(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = dec(x)

            return torch.sigmoid(self.head(x))

    return UNet


try:
    import torch  # noqa: F401
    UNet = _build_unet_class()
except ImportError:
    pass
