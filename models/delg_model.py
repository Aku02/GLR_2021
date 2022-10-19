import timm
from torch import nn
import torch
from torch.nn import functional as F
from torch import nn
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np

import sys
sys.path.append("/mount/delg")
from model.resnet import ResNet, ResHead
from model.delg_model import SpatialAttention2d, Arcface

import core.config as config
from core.config import cfg as delg_cfg

config.load_cfg('/mount/delg/configs/',cfg_dest='resnet_delg_8gpu.yaml')
config.assert_and_infer_cfg()
delg_cfg.freeze()

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.cfg = cfg
        self.globalmodel = ResNet()
        self.desc_cls = Arcface(delg_cfg.MODEL.HEADS.REDUCTION_DIM, delg_cfg.MODEL.NUM_CLASSES)
        self.localmodel = SpatialAttention2d(1024)
        self.att_cls = ResHead(512, delg_cfg.MODEL.NUM_CLASSES)
    
    def forward(self, batch):
        
        x = batch['input']
        targets = batch['target']
        
        global_feature, feamap = self.globalmodel(x)
        
        block3 = feamap.detach()
        local_feature, att_score = self.localmodel(block3)
        local_logits = self.att_cls(local_feature)
        
        if self.cfg.headless:
            return {"target": batch['target'],'embeddings': global_feature}
        
        global_logits = self.desc_cls(global_feature, targets)
        return global_feature, global_logits, local_feature, local_logits, att_score

# class Net(nn.Module):
#     def __init__(self, cfg):
#         super(Net, self).__init__()

#         self.cfg = cfg
#         self.n_classes = self.cfg.n_classes
#         self.backbone = Delg()


#         self.embedding_size = cfg.embedding_size


#     def forward(self, batch):

#         x = batch['input']

#         x = self.backbone(x)
#         x = self.global_pool(x)
#         x = x[:,:,0,0]

#         x_emb = self.neck(x)

#         if self.cfg.headless:
#             return {"target": batch['target'],'embeddings': x_emb}
        
#         logits = self.head(x_emb)
# #         loss = self.loss_fn(logits, batch['target'].long(), self.n_classes)
#         preds = logits.softmax(1)
#         preds_conf, preds_cls = preds.max(1)
#         if self.training:
#             loss = self.loss_fn(logits, batch['target'].long())
#             return {'loss': loss, "target": batch['target'], "preds_conf":preds_conf,'preds_cls':preds_cls}
#         else:
#             loss = torch.zeros((1),device=x.device)
#             return {'loss': loss, "target": batch['target'],"preds_conf":preds_conf,'preds_cls':preds_cls,
#                     'embeddings': x_emb
#                    }
