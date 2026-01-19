import torch.nn as nn

from . import register_model


_VGG_CFG = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class _VGG(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super().__init__()
        self.features = self._make_layers(cfg)
        # Flatten size after adaptive pooling will be 512
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        # Use adaptive pooling to (1,1) regardless of spatial input size
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


@register_model("vgg11")
def vgg11(num_classes):
    return _VGG(_VGG_CFG["vgg11"], num_classes)


@register_model("vgg13")
def vgg13(num_classes):
    return _VGG(_VGG_CFG["vgg13"], num_classes)


@register_model("vgg16")
def vgg16(num_classes):
    return _VGG(_VGG_CFG["vgg16"], num_classes)


@register_model("vgg19")
def vgg19(num_classes):
    return _VGG(_VGG_CFG["vgg19"], num_classes)


