import torchvision


def get_model(backbone_name, num_classes, **kwargs):
    if 'resnet50' in backbone_name:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    if 'resnet101' in backbone_name:
        backbone = resnet_fpn_backbone('resnet101', False)
        model = FasterRCNN(backbone, 91, **kwargs)

        #state_dict = load_state_dict_from_url('https://ababino-models.s3.amazonaws.com/resnet101_7a82fa4a.pth', map_location='cpu')
        #model.load_state_dict(state_dict['model'])
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    if 'resnet' in backbone_name:
        norm_layer = misc_nn_ops.FrozenBatchNorm2d
    else:
        norm_layer= None
    backbone = resnet.__dict__[backbone_name](
        pretrained=True, norm_layer=norm_layer)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    backbone =  BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

    return FasterRCNN(backbone, num_classes, **kwargs)
