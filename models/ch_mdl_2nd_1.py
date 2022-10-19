from efficientnet_pytorch import EfficientNet
from torch import nn
# model = EfficientNet.from_pretrained('efficientnet-b0')

import torch.nn.functional as F
# --------------------------------------
# Pooling layers
# --------------------------------------
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class Net(nn.Module):

    def __init__(self,cfg, ds=None):
        super().__init__()
        num_classes = cfg.n_classes
        self.cfg = cfg
        if cfg.backbone == 'efficientnet_b5':
            self.backbone = EfficientNet.from_name('efficientnet-b5')
            feat_dim = 2048
        elif cfg.backbone == 'efficientnet_b6':
            self.backbone = EfficientNet.from_name('efficientnet-b6')
            feat_dim = 2304
        elif cfg.backbone == 'efficientnet_b7':
            self.backbone = EfficientNet.from_name('efficientnet-b7')
            feat_dim = 2560

        self.in_channels = 3
        self.pool = gem
        fc_dim = 512
        self.fc = nn.Linear(feat_dim, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
#         if loss_module == 'arcface':
#           self.face_margin_product = ArcMarginProduct(fc_dim, num_classes, s=s, m=margin)
#         elif loss_module == 'arcface2':
#           self.face_margin_product = ArcMarginProduct2(fc_dim, num_classes, s=s, m=margin)
#         else:
#           raise ValueError(loss_module)

    def forward(self, batch):
        x = batch['input']
        x = self.backbone.extract_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x_emb = self.bn(x)
        if self.cfg.headless:
            return {"target": batch['target'],'embeddings': x_emb}
