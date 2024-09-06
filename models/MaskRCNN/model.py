import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import faster_rcnn


def get_maskrcnn_resnet50_fpn_v2(n, weights_path=None):
    maskrcnn = maskrcnn_resnet50_fpn_v2()
    if weights_path is not None:
        maskrcnn.load_state_dict(torch.load(weights_path, weights_only=True))
    for parameter in maskrcnn.parameters():
        parameter.requires_grad = False

    cin = maskrcnn.roi_heads.box_predictor.cls_score.in_features
    maskrcnn.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(cin, num_classes=n)

    return maskrcnn
