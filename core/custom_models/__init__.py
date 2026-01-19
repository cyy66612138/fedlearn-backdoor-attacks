# Simple local registry for models
_MODEL_REGISTRY = {}


def register_model(name):
    def decorator(obj):
        _MODEL_REGISTRY[name.lower()] = obj
        return obj
    return decorator


def all_models():
    return list(_MODEL_REGISTRY.keys())


def get_model(model_name, num_channels=3, num_classes=10, num_dims=32, device=None):
    name = model_name.lower()
    if name not in _MODEL_REGISTRY:
        raise NotImplementedError(f"Model not implemented, choose from {list(_MODEL_REGISTRY.keys())}")

    grey_only = {"lr"}
    adaptive = {"lenet", "lenet_bn"}
    rgb_only = {
        # Original ResNet models
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        # # VGG models
        # "vgg11", "vgg13", "vgg16", "vgg19"
    }

    if name in grey_only:
        assert num_channels == 1, "This model supports only 1-channel inputs"
        model = _MODEL_REGISTRY[name](input_dim=num_dims * num_dims, num_classes=num_classes)
    elif name in adaptive:
        model = _MODEL_REGISTRY[name](num_channels=num_channels, num_classes=num_classes)
    elif name in rgb_only:
        assert num_channels == 3, "This model supports only 3-channel inputs"
        model = _MODEL_REGISTRY[name](num_classes=num_classes)
    elif name == "simplecnn":
        model = _MODEL_REGISTRY[name](input_size=(num_channels, num_dims, num_dims), num_classes=num_classes)
    else:
        model = _MODEL_REGISTRY[name](num_classes=num_classes)

    if device is not None:
        return model.to(device)
    # fallback to current default device
    try:
        default_device = next(model.parameters()).device
        return model.to(default_device)
    except StopIteration:
        return model


# Import modules to populate registry
from . import lenet  # noqa: F401
from . import simplecnn  # noqa: F401
from . import resnet  # noqa: F401