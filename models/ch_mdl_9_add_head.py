import timm
from torch import nn
import torch
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, n_dim, ls_=0.9):
        super().__init__()
        self.n_dim = n_dim
        self.ls_ = ls_

    def forward(self, x, target):
        target = F.one_hot(target, self.n_dim).float()
        target *= self.ls_
        target += (1 - self.ls_) / self.n_dim

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, n_classes, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
        self.out_dim =n_classes
            
    def forward(self, logits, labels):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss     



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="bce", weight=None, reduction="mean",class_weights_norm=None ):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm
        
        if crit == "focal":
            self.crit = FocalLoss(gamma=args.focal_loss_gamma)
        elif crit == "bce":
            self.crit = nn.CrossEntropyLoss(reduction="none")   
        elif crit == "label_smoothing":
            self.crit = LabelSmoothingLoss(classes=args.n_classes)   

        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):
        #print(self.weight[labels])
        #print(self.s)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

#         labels2 = torch.nn.functional.one_hot(labels, num_classes=args.n_classes+2)
#         labels2 = labels2[:,:args.n_classes+1]
        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)

            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()
            
            return loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss    

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)   
        return ret
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()      

class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()
        
        self.d6 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[0],padding='same')
        self.d12 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[1],padding='same')
        self.d18 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[2],padding='same')
        self.conv1 = nn.Conv2d(512 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        
        x6 = self.d6(x)
        x12 = self.d12(x)
        x18 = self.d18(x)
        x = torch.cat([x6,x12,x18],dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return x

class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        
        x = att * feature_map_norm
        return x, att_score   

class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):

        bs, c, w, h = fl.shape
        
        fl_dot_fg = torch.bmm(fg[:,None,:],fl.reshape(bs,c,-1))
        fl_dot_fg = fl_dot_fg.reshape(bs,1,w,h)
        fg_norm = torch.norm(fg, dim=1)
        
        fl_proj = (fl_dot_fg / fg_norm[:,None,None,None]) * fg[:,:,None,None]
        fl_orth = fl - fl_proj
        
        f_fused = torch.cat([fl_orth,fg[:,:,None,None].repeat(1,1,w,h)],dim=1)
        return f_fused  


def complexity_conv2d(cx, w_in, w_out, k, stride, padding, groups=1, bias=False):
    """Accumulates complexity of Conv2D into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h = (h + 2 * padding - k) // stride + 1
    w = (w + 2 * padding - k) // stride + 1
    flops += k * k * w_in * w_out * h * w // groups
    params += k * k * w_in * w_out // groups
    flops += w_out if bias else 0
    params += w_out if bias else 0
    acts += w_out * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}    

class ResHead(nn.Module):
    """ResNet head: AvgPool, 1x1."""

    def __init__(self, w_in, nc):
        super(ResHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, nc):
        cx["h"], cx["w"] = 1, 1
        cx = complexity_conv2d(cx, w_in, nc, 1, 1, 0, bias=True)
        return cx

class Net(nn.Module):
    def __init__(self, cfg, dataset):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = self.cfg.n_classes
        self.backbone = timm.create_model(cfg.backbone, 
                                          pretrained=cfg.pretrained, 
                                          num_classes=0, 
                                          global_pool="", 
                                          in_chans=self.cfg.in_channels,features_only = True)

        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']
        
        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g

        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = cfg.embedding_size

        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if cfg.neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(fusion_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif cfg.neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(fusion_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif cfg.neck == "option-X":
            self.neck = nn.Sequential(
                nn.Linear(fusion_out, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )
            
        elif cfg.neck == "option-S":
            self.neck = nn.Sequential(
                nn.Linear(fusion_out, self.embedding_size),
                Swish_module()
            )

        if not self.cfg.headless:    
            self.head_in_units = self.embedding_size
            self.head = ArcMarginProduct_subcenter(self.embedding_size, self.n_classes)
            self.head2 = ArcMarginProduct_subcenter(self.embedding_size, self.cfg.n_classes2)
        if self.cfg.loss == 'adaptive_arcface':
            self.loss_fn = ArcFaceLossAdaptiveMargin(dataset.margins,self.n_classes+self.cfg.n_classes2,cfg.arcface_s)
        elif self.cfg.loss == 'arcface':
            self.loss_fn = ArcFaceLoss(cfg.arcface_s,cfg.arcface_m)
        else:
            pass
        
        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, self.cfg.dilations)
        self.conv_g = nn.Conv2d(backbone_out,feature_dim_l_g,kernel_size=1)
        self.bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g =  nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()

        if cfg.freeze_backbone_head:
            for name, param in self.named_parameters():
                if not 'head.' in name:
                    param.requires_grad = False
        
    def forward(self, batch, scale=1):

        x = batch['input']
        
        dev = x.device
        if scale != 1:
            x = F.interpolate(x, scale_factor=scale)

        x = self.backbone(x)
        
        x_l = x[-2]
        x_g = x[-1]
        
        x_l = self.mam(x_l)
        x_l, att_score = self.attention2d(x_l)
        
        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)
        
        x_g = self.global_pool(x_g)
        x_g = x_g[:,:,0,0]
        
        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:,:,0,0]        
        
        x_emb = self.neck(x_fused)

        if self.cfg.headless:
            return {"target": batch['target'],'embeddings': x_emb}

        
        logits = self.head(x_emb)
        logits2 = self.head2(x_emb)
        logits = torch.cat([logits,logits2],dim=-1)
        
#         loss = self.loss_fn(logits, batch['target'].long(), self.n_classes)
        preds = logits.softmax(1)
        preds_conf, preds_cls = preds.max(1)
        if self.training:
            loss = self.loss_fn(logits, batch['target'].long())

            return {'loss': loss, "target": batch['target'], "preds_conf":preds_conf,'preds_cls':preds_cls}

        else:
            loss = torch.zeros((1),device=dev)
            return {'loss': loss, "target": batch['target'],"preds_conf":preds_conf,'preds_cls':preds_cls,
                    'embeddings': x_emb
                   }

    def freeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False


    def unfreeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = True
