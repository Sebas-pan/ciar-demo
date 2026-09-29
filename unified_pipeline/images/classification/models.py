"""Image classification models (torchvision).

Supported `model.type` values:
- vgg16 / vgg16_bn (classifier head)
- resnet50 / resnet18 / resnet101
- efficientnet_b0, mobilenet_v3_large, densenet121
- vit_b_16, swin_t, convnext_tiny
"""
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


def _finetune_all(model):
    for p in model.parameters():
        p.requires_grad = True
    return model


def create_resnet(cfg, num_classes, depth=50):
    from torchvision.models import (
        resnet18, resnet50, resnet101,
        ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
    )
    model_cfg = cfg.get("model", {})
    pretrained = bool(model_cfg.get("pretrained", True))
    if int(depth) == 18:
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif int(depth) == 101:
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet101(weights=weights)
    else:
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    return _finetune_all(model)


def create_resnet50(cfg, num_classes):
    return create_resnet(cfg, num_classes, depth=50)


def create_efficientnet_b0(cfg, num_classes):
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    pretrained = bool(cfg.get("model", {}).get("pretrained", True))
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, int(num_classes))
    return _finetune_all(model)


def create_mobilenet_v3(cfg, num_classes):
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    pretrained = bool(cfg.get("model", {}).get("pretrained", True))
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, int(num_classes))
    return _finetune_all(model)


def create_densenet121(cfg, num_classes):
    from torchvision.models import densenet121, DenseNet121_Weights
    pretrained = bool(cfg.get("model", {}).get("pretrained", True))
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, int(num_classes))
    return _finetune_all(model)


def create_vit_b16(cfg, num_classes):
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    pretrained = bool(cfg.get("model", {}).get("pretrained", True))
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, int(num_classes))
    return _finetune_all(model)


def create_swin_t(cfg, num_classes):
    from torchvision.models import swin_t, Swin_T_Weights
    pretrained = bool(cfg.get("model", {}).get("pretrained", True))
    weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, int(num_classes))
    return _finetune_all(model)


def create_convnext_tiny(cfg, num_classes):
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    pretrained = bool(cfg.get("model", {}).get("pretrained", True))
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, int(num_classes))
    return _finetune_all(model)


def create_classifier(cfg, num_classes):
    """Generic factory — dispatches on model.type / variant."""
    model_cfg = cfg.get("model", {})
    key = str(model_cfg.get("type", model_cfg.get("variant", "vgg16"))).lower().replace("-", "_")
    if key in ("resnet50", "resnet", "resnet_50"):
        return create_resnet(cfg, num_classes, depth=50)
    if key in ("resnet18", "resnet_18"):
        return create_resnet(cfg, num_classes, depth=18)
    if key in ("resnet101", "resnet_101"):
        return create_resnet(cfg, num_classes, depth=101)
    if key in ("efficientnet_b0", "efficientnet", "efficientnetb0"):
        return create_efficientnet_b0(cfg, num_classes)
    if key in ("mobilenet_v3_large", "mobilenet_v3", "mobilenet", "mobilenetv3"):
        return create_mobilenet_v3(cfg, num_classes)
    if key in ("densenet121", "densenet", "densenet_121"):
        return create_densenet121(cfg, num_classes)
    if key in ("vit_b_16", "vit", "vitb16", "vit_b16"):
        return create_vit_b16(cfg, num_classes)
    if key in ("swin_t", "swin", "swint", "swin_tiny"):
        return create_swin_t(cfg, num_classes)
    if key in ("convnext_tiny", "convnext", "convnexttiny"):
        return create_convnext_tiny(cfg, num_classes)
    return create_vgg16(cfg, num_classes)


def identify_model_type(model_type="vgg16"):
    key = str(model_type or "vgg16").lower().replace("-", "_")
    if key.startswith("resnet"):
        return "image_resnet_classification"
    if key.startswith(("efficientnet", "mobilenet", "densenet")):
        return "image_cnn_classification"
    if key.startswith(("vit", "swin", "convnext")):
        return "image_transformer_classification"
    return "image_vgg_classification"
