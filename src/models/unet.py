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
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        self.deconv  = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5) if dropout else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.deconv(x)))
        return self.dropout(x)

class UNet(nn.Module):
    def __init__(self, base_channels=16, depth=6):
        super().__init__()

        chs = [1] + [base_channels * 2**i for i in range(depth)]
        self.encoders = nn.ModuleList([
            EncoderBlock(chs[i], chs[i+1]) for i in range(depth)
        ])

        self.decoders = nn.ModuleList()
        for i in range(depth):
            in_ch = chs[depth - i] if i == 0 else chs[depth - i] * 2
            out_ch = chs[depth - i - 1]
            dropout = i < 3  # 50% dropout sur les 3 premières couches du décodeur
            self.decoders.append(DecoderBlock(in_ch, out_ch, dropout=dropout))

        self.final = nn.Sequential(
            nn.Conv2d(chs[0], 1, kernel_size=1),
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
    model = UNet(base_channels=16, depth=6)
    dummy = torch.randn(2, 1, 512, 128)  # (batch, canal, fréquences, temps)
    out = model(dummy)
    print(out.shape)
    print(out.min().item(), out.max().item())
