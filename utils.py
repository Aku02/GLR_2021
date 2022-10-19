from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import AdamW
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DataLoader
import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch import nn, optim
from boto3.session import Session
import boto3

# from torch.cuda.amp import GradScaler, autocast
# from torch.nn.parallel import DistributedDataParallel as NativeDDP
import importlib
import math
import neptune.new as neptune
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class OrderedDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        print("TOTAL SIZE", self.total_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self.rank * self.num_samples : self.rank * self.num_samples + self.num_samples
        ]
        print(
            "SAMPLES",
            self.rank * self.num_samples,
            self.rank * self.num_samples + self.num_samples,
        )
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def sync_across_gpus(t, world_size):
    torch.distributed.barrier()
    gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor)


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_model(cfg, ds):
    Net = importlib.import_module(cfg.model).Net
    if cfg.loss == 'adaptive_arcface':
        net = Net(cfg,ds)
    else:
        net = Net(cfg)
    if cfg.pretrained_weights is not None:
        print(f'{cfg.local_rank}: loading weights from',cfg.pretrained_weights)
        state_dict = torch.load(cfg.pretrained_weights, map_location='cpu')['model']
        state_dict = {key.replace('module.',''):val for key,val in state_dict.items()}
        if cfg.pop_weights is not None:
            print(f'popping {cfg.pop_weights}')
            to_pop = []
            for key in state_dict:
                for item in cfg.pop_weights:
                    if item in key:
                        to_pop += [key]
            for key in to_pop:
                print(f'popping {key}')
                state_dict.pop(key)

        net.load_state_dict(state_dict, strict=cfg.pretrained_weights_strict)
        print(f'{cfg.local_rank}: weights loaded from',cfg.pretrained_weights)
    
    return net


def create_checkpoint(cfg, model, optimizer, epoch, scheduler=None, scaler=None, save_headless=False):

    
    state_dict = model.state_dict()
    if save_headless:
        to_pop = [item for item in state_dict if '.head.' in item]
        for item in to_pop:
            state_dict.pop(item)
    
    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

def load_checkpoint(cfg, model, optimizer, scheduler=None, scaler=None):
    
    print(f'loading ckpt {cfg.resume_from}')
    checkpoint = torch.load(cfg.resume_from, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler_dict = checkpoint['scheduler']
    if scaler is not None:    
        scaler.load_state_dict(checkpoint['scaler'])
        
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler_dict, scaler, epoch

def get_dataset(df, cfg, mode='train',allowed_targets=None):
    
    #modes train, val, index
    print(f"Loading {mode} dataset")
    
    if mode == 'train':
        ds_mode = 'train'
        aug = cfg.train_aug
    elif mode == 'train_val':
        aug = cfg.val_aug
        ds_mode = 'train'
    else:
        aug = cfg.val_aug
        ds_mode = mode
    
    dataset = cfg.CustomDataset(df, cfg, aug=aug, mode=ds_mode,allowed_targets=allowed_targets)
    return dataset
    

def get_train_dataset(train_df, cfg,allowed_targets=None):
    print("Loading train dataset")

    train_dataset = cfg.CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train",allowed_targets=allowed_targets)
    return train_dataset

def get_dataloader(ds, cfg, mode='train'):
    
    if mode == 'train':
        dl = get_train_dataloader(ds, cfg)
    elif mode =='val':
        dl = get_val_dataloader(ds, cfg)
    return dl

def get_train_dataloader(train_ds, cfg):

    if cfg.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=cfg.world_size, rank=cfg.local_rank, shuffle=True, seed=cfg.seed
        )
    else:
        sampler = None

    train_dataloader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataset(val_df, cfg, allowed_targets=None):
    print("Loading val dataset")
    val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val",allowed_targets=allowed_targets)
    return val_dataset

# def get_val_index_dataset(train_df, train_dataset):
#     print("Loading val dataset")
#     val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")
#     return val_dataset

def get_val_dataloader(val_ds, cfg):

    if cfg.distributed and cfg.eval_ddp:
        sampler = OrderedDistributedSampler(
            val_ds, num_replicas=cfg.world_size, rank=cfg.local_rank
        )
    else:
        sampler = SequentialSampler(val_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_test_dataset(test_df, cfg):
    print("Loading test dataset")
    test_dataset = cfg.CustomDataset(test_df, cfg, aug=cfg.val_aug, mode="test")
    return test_dataset


def get_test_dataloader(test_ds, cfg):

    if cfg.distributed and cfg.eval_ddp:
        sampler = OrderedDistributedSampler(
            test_ds, num_replicas=cfg.world_size, rank=cfg.local_rank
        )
    else:
        sampler = SequentialSampler(test_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    test_dataloader = DataLoader(
        test_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"test: dataset {len(test_ds)}, dataloader {len(test_dataloader)}")
    return test_dataloader


def get_optimizer(model, cfg):

    # params = [{"params": [param for name, param in model.named_parameters()], "lr": cfg.lr,"weight_decay":cfg.weight_decay}]
    params = model.parameters()

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "AdamW":
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.sgd_momentum,
            nesterov=cfg.sgd_nesterov,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "fused_SGD":
        import apex

        optimizer = apex.optimizers.FusedSGD(
            params, lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "fused_Adam":
        import apex

        optimizer = apex.optimizers.FusedAdam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "SGD_AGC":
        from nfnets import SGD_AGC

        optimizer = SGD_AGC(
            named_params=model.named_parameters(),  # Pass named parameters
            lr=cfg.lr,
            momentum=0.9,
            clipping=0.1,  # New clipping parameter
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )

    return optimizer


def get_scheduler(cfg, optimizer, total_steps):

    if cfg.schedule == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs_step * (total_steps // cfg.batch_size) // cfg.world_size,
            gamma=0.5,
        )
    elif cfg.schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size) // cfg.world_size,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
        )
    elif cfg.schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
        )

        print("num_steps", (total_steps // cfg.batch_size) // cfg.world_size)

    else:
        scheduler = None

    return scheduler


def setup_neptune(cfg):
    neptune_run = neptune.init(
        project=cfg.neptune_project,
        tags=cfg.tags,
        mode=cfg.neptune_connection_mode,
        capture_stdout=False,
        capture_stderr=False,
        source_files=[f'models/{cfg.model}.py',f'data/{cfg.dataset}.py']
    )


    neptune_run["cfg"] = cfg.__dict__

    return neptune_run


def get_data(cfg):

    # setup dataset

    print(f"reading {cfg.train_df}")
    train_df = pd.read_csv(cfg.train_df)

    if cfg.data_sample > -1:
        train_df = train_df.sample(cfg.data_sample)

    if cfg.val_df is not None:
        print(f"reading {cfg.val_df}")
        val_df = pd.read_csv(cfg.val_df)

    if cfg.test:
        test_df = pd.read_csv(cfg.test_df)
    else:
        test_df = None
        
        
    return train_df, val_df, test_df

def get_data_retrieval(cfg):
    
    print(f"reading {cfg.query_df}")
    query_df = pd.read_csv(cfg.query_df)
    index_df = pd.read_csv(cfg.index_df)
        
    return query_df, index_df    


def save_first_batch(feature_dict, cfg):
    print("Saving first batch of images")
    images = feature_dict["input"].detach().cpu().numpy()
    targets = feature_dict["target"].detach().cpu().numpy()
    boxes_batch = feature_dict["boxes"]

    for i, (image, target, boxes) in enumerate(zip(images, targets, boxes_batch)):
        fig, ax = plt.subplots(figsize=(13, 13))
        print(f"image_{i}: min {image[0].min()}, max {image[0].max()}")
        ax.imshow(image[0])  # just one channel / greyscale
        boxes = boxes.detach().cpu().numpy()
        for ii in range(len(boxes)):
            w = boxes[ii, 2] - boxes[ii, 0]
            h = boxes[ii, 3] - boxes[ii, 1]
            rect = patches.Rectangle((boxes[ii, 1], boxes[ii, 0]), h, w, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        fig.suptitle(f"Target: {target}")
        fig.savefig(f"{cfg.output_dir}/fold{cfg.fold}/batch1_image{i}_seed{cfg.seed}.png")
        plt.close()


def save_first_batch_preds(feature_dict, output_dict, cfg):
    print("Saving preds of first batch of images")
    images = feature_dict["input"].detach().cpu().numpy()
    targets = feature_dict["target"].detach().cpu().numpy()
    class_preds = output_dict["class_logits"].softmax(1).detach().cpu().numpy()
    boxes_batch = feature_dict["boxes"]
    pred_boxes_batch = output_dict["detections"].detach().cpu().numpy()
    # pred_boxes_batch = pred_boxes_batch[:, :, 4]
    # print(pred_boxes_batch.shape)

    for i, (image, boxes, boxes_pred) in enumerate(zip(images, boxes_batch, pred_boxes_batch)):
        fig, ax = plt.subplots(figsize=(13, 13))
        ax.imshow(image[0])  # just one channel / greyscale
        boxes = boxes.detach().cpu().numpy()
        for ii in range(len(boxes)):
            w = boxes[ii, 2] - boxes[ii, 0]
            h = boxes[ii, 3] - boxes[ii, 1]
            rect = patches.Rectangle((boxes[ii, 1], boxes[ii, 0]), h, w, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        for ii in range(len(boxes_pred)):
            if boxes_pred[ii, 4] > 0.3:
                w = boxes_pred[ii, 2] - boxes_pred[ii, 0]
                h = boxes_pred[ii, 3] - boxes_pred[ii, 1]
                rect = patches.Rectangle((boxes_pred[ii, 0], boxes_pred[ii, 1]), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(boxes_pred[ii, 0] + w, boxes_pred[ii, 1] + h, f"{boxes_pred[ii, 4]}")
        fig.suptitle(f"Target: {targets[i]}, Preds: {class_preds[i]}")
        fig.savefig(f"{cfg.output_dir}/fold{cfg.fold}/preds_batch1_image{i}_seed{cfg.seed}.png")
        plt.close()