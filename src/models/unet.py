import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    """Transposed conv (+ BN + ReLU unless `return_logit`) (+ dropout). With `return_logit` the block outputs the raw
    deconvolution, so the 1x1 conv + sigmoid that follow see an unbounded signal (see `UNet(legacy_head=...)`)."""
    def __init__(self, in_ch, out_ch, dropout=False, return_logit=False):
        super().__init__()
        self.return_logit = return_logit
        self.deconv  = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2, output_padding=1)
        if not return_logit:
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5) if dropout else nn.Identity()

    def forward(self, x):
        if self.return_logit:
            return self.deconv(x)
        x = self.act(self.bn(self.deconv(x)))
        return self.dropout(x)

class UNet(nn.Module):
    """`legacy_head=True` restores BN + ReLU in the last decoder block, as before commit 68ba08f. In that block
    (`in_channels` output channels) the BN rescales and the ReLU clips the head input to >= 0 before the 1x1 conv + sigmoid:
    the bug that commit fixed. Kept only to reproduce the pre-fix runs and to measure the bug's effect (same state_dict keys
    as the pre-fix model, so old checkpoints load with `legacy_head=True`)."""
    def __init__(self, base_channels: int, depth: int, in_channels: int = 1, legacy_head: bool = False):
        super().__init__()

        chs = [in_channels] + [base_channels * 2**i for i in range(depth)]
        self.encoders = nn.ModuleList([
            EncoderBlock(chs[i], chs[i+1]) for i in range(depth)
        ])

        self.decoders = nn.ModuleList()
        for i in range(depth):
            in_ch = chs[depth - i] if i == 0 else chs[depth - i] * 2
            out_ch = chs[depth - i - 1]
            dropout = i < 3  # 50% dropout on decoder blocks 1, 2, 3 (Jansson et al. 2017)
            return_logit = i == depth - 1 and not legacy_head
            self.decoders.append(DecoderBlock(in_ch, out_ch, dropout=dropout, return_logit=return_logit))

        self.final = nn.Sequential(
            nn.Conv2d(chs[0], in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** len(self.encoders)
        x = F.pad(x, (0, -W % m, 0, -H % m))

        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        skips = skips[:-1][::-1]

        for i, dec in enumerate(self.decoders):
            x = dec(x)
            if i < len(skips):
                x = torch.cat([x, skips[i]], dim=1)

        return self.final(x)[..., :H, :W]

if __name__ == "__main__":
    model = UNet(base_channels=16, depth=6, in_channels=2)
    dummy = torch.randn(2, 2, 512, 128)  # (batch, channel, frequency, time)
    out = model(dummy)
    print(out.shape)
    print(out.min().item(), out.max().item())
