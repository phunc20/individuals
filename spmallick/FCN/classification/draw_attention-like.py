import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision import models, transforms
from torch.hub import load_state_dict_from_url
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

DARK_READER = True
if DARK_READER:
    plt.rcParams.update({
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": "white",
        "axes.titlecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "lightgray",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black",
    })


class FCNResnet18(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
        super().__init__(
            block=models.resnet.BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
            **kwargs
        )
        if pretrained:
            state_dict = load_state_dict_from_url(models.resnet.model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py#L153-L154
        self.avgpool = torch.nn.AvgPool2d((7, 7))

        self.last_conv = torch.nn.Conv2d(
            in_channels=self.fc.in_features,
            out_channels=num_classes,
            kernel_size=1,
        )
        self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)

    # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py#L197-L213
    #def _forward_impl(self, x):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = self.last_conv(x)
        return x

def imshowBGR(ndarray):
    plt.imshow(ndarray[...,::-1]);


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image",
        help="Path to the image to be inferred",
        default="otter.jpg",
    )
    args = parser.parse_args()
    path_image = Path(args.image)

    model = FCNResnet18(pretrained=True).eval()
    original_image = cv2.imread(str(path_image))
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # add batch size = 1

    with torch.no_grad():
        preds = model(image)

    preds = torch.softmax(preds, dim=1)
    pred, class_idx = torch.max(preds, dim=1)
    row_max, row_idx = torch.max(pred, dim=1)
    col_max, col_idx = torch.max(row_max, dim=1)

    with open("imagenet_classes.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    pred_class_id = class_idx[0, row_idx[0, col_idx], col_idx]
    pred_class = labels[pred_class_id]

    score_map = preds[0, pred_class_id, :, :].cpu().numpy()
    score_map = score_map[0]
    h, w = original_image.shape[:2]
    score_map = cv2.resize(score_map, (w, h))


    # Apply score map as a mask to original image
    #score_map = score_map - np.min(score_map[:])
    #score_map = score_map / np.max(score_map[:])
    score_map = score_map - score_map.min()
    score_map = score_map / score_map.max()

    score_map = cv2.cvtColor(score_map, cv2.COLOR_GRAY2BGR)
    masked_image = (original_image * score_map).astype(np.uint8)

    #imshowBGR(score_map)
    imshowBGR(masked_image)
    plt.title(f"pred_class = {pred_class}")
    plt.show()

    #_, score_map_for_contours = cv2.threshold(
    #    score_map,
    #    0.25,
    #    1,
    #    type=cv2.THRESH_BINARY,
    #)

    #score_map_for_contours = score_map_for_contours.astype(np.uint8)

    #contours, _ = cv2.findContours(
    #    score_map_for_contours,
    #    mode=cv2.RETR_EXTERNAL,
    #    method=cv2.CHAIN_APPROX_SIMPLE,
    #)

    #rect = cv2.boundingRect(contours[0])
    #red = (0,0,255)
    #cv2.rectangle(
    #    masked_image,
    #    rect[:2],
    #    (rect[0] + rect[2], rect[1] + rect[3]),
    #    red,
    #    2,
    #)

    #imshowBGR(masked_image)
    #plt.title(f"pred_class = {pred_class}")
    #plt.show()


