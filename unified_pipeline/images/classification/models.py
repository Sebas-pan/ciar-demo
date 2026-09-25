"""VGG16 / ResNet50 classification models."""
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


def create_resnet50(cfg, num_classes):
    from torchvision.models import resnet50, ResNet50_Weights
    model_cfg = cfg.get("model", {})
    pretrained = bool(model_cfg.get("pretrained", True))

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    # Replace classifier head (full fine-tune: all params trainable)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, int(num_classes))
    for p in model.parameters():
        p.requires_grad = True
    return model


def create_classifier(cfg, num_classes):
    """Generic factory — dispatches on model.type / variant."""
    model_cfg = cfg.get("model", {})
    key = str(model_cfg.get("type", model_cfg.get("variant", "vgg16"))).lower()
    if key in ("resnet50", "resnet", "resnet-50", "resnet_50"):
        return create_resnet50(cfg, num_classes)
    return create_vgg16(cfg, num_classes)


def identify_model_type(model_type="vgg16"):
    key = str(model_type or "vgg16").lower()
    if key in ("resnet50", "resnet", "resnet-50", "resnet_50"):
        return "image_resnet_classification"
    return "image_vgg_classification"
