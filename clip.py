import numpy as np
import torch
from torch import nn


class CLIP(nn.Module):
    def __init__(self, encode_image, encode_text, temperature):
        super().__init__()

        self.encode_image = encode_image
        self.encode_text = encode_text
        self.image_projection = nn.Linear(2048, 768)
        self.text_projection = nn.Linear(768, 768)

        self.logit_scale = nn.Parameter(torch.tensor(temperature))# * np.log(1 / 0.0001))

    def forward(self, x):
        img, text = x['image'], x['text']

        image_features = self.encode_image(img)
        text_features = self.encode_text(text).pooler_output

        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_images = logit_scale * image_features @ text_features.t()

        return logits_per_images

