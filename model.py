import torch.nn as nn
import torch.nn.functional as F


class BaselineEmbeddingNet(nn.Module):
    def __init__(self):
        super(BaselineEmbeddingNet, self).__init__()
        self.fully_conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, groups=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, groups=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=2, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 32, kernel_size=3, stride=1, groups=2, bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_net = BaselineEmbeddingNet()
        self.match_batchnorm = nn.BatchNorm2d(1)

    def forward(self, template, search):
        embedding_reference = self.embedding_net(template)
        embedding_search = self.embedding_net(search)
        match_map = self.match_corr(embedding_reference, embedding_search)
        return match_map

    def match_corr(self, embed_ref, embed_srch):
        b, c, h, w = embed_srch.shape
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w), embed_ref, groups=b)
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)
        return match_map