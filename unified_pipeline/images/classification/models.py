"""VGG16 classification model."""
import torch
import torch.nn as nn


def create_vgg16(cfg, num_classes):
    from torchvision.models import vgg16, vgg16_bn, VGG16_Weights, VGG16_BN_Weights
    model_cfg = cfg.get("model", {})
    variant = str(model_cfg.get("variant", "vgg16")).lower()
    pretrained = bool(model_cfg.get("pretrained", True))
    dropout = float(model_cfg.get("dropout", 0.5))

    if variant in ("vgg16_bn", "vgg16-bn", "vgg_bn"):
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg16_bn(weights=weights)
    else:
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg16(weights=weights)

    # Replace classifier head
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, int(num_classes))
    # Adjust dropout if requested (classifier has dropout at 0,2)
    if dropout != 0.5:
        for i in [0, 2]:
            if isinstance(model.classifier[i], nn.Dropout):
                model.classifier[i].p = dropout
    return model


def identify_model_type():
    return "image_vgg_classification"
