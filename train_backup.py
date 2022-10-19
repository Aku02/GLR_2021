import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm
import gc
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.utils.agc import adaptive_clip_grad
from collections import defaultdict

from utils import (
    sync_across_gpus,
    set_seed,
    get_model,
    create_checkpoint,
    get_train_dataset,
    get_data,
)
from utils import (
    get_train_dataloader,
    get_val_dataset,
    get_val_dataloader,
    get_test_dataset,
    get_test_dataloader,
    get_optimizer,
    get_scheduler,
    setup_neptune,
    save_first_batch,
    save_first_batch_preds,
    upload_s3
)

import cv2
from metrics import calc_metric, comp_metric
from copy import copy
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


cv2.setNumThreads(0)


sys.path.append("configs")
sys.path.append("models")
sys.path.append("data")
sys.path.append("losses")
sys.path.append("utils")


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)

# overwrite params in config with additional args
if len(other_args) > 1:
    other_args = {k.replace('-',''):v for k, v in zip(other_args[1::2], other_args[2::2])}
    
    for key in other_args:
        if key in cfg.__dict__: 
            
            print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')
            cfg_type = type(cfg.__dict__[key])
            if cfg_type == bool:
                cfg.__dict__[key] = other_args[key] == 'True'
            elif cfg_type == type(None):
                cfg.__dict__[key] = other_args[key]
            else:
                cfg.__dict__[key] = cfg_type(other_args[key])

if cfg.test:
    from submission import create_submission


os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)

cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
cfg.tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
cfg.val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device

def get_preds(model, val_dataloader, cfg, pre="val"):
    saved_images = False
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    val_data = defaultdict(list)

    for data in tqdm(val_dataloader, disable=cfg.local_rank != 0):

        batch = batch_to_device(data, device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        if (not saved_images) & (cfg.save_first_batch_preds):
            save_first_batch_preds(batch, output, cfg)
            saved_images = True

        for key, val in output.items():
            val_data[key] += [output[key]]

    for key, val in output.items():
        value = val_data[key]
        if isinstance(value[0], list):
            val_data[key] = [item for sublist in value for item in sublist]
        
        else:
            if len(value[0].shape) == 0:
                val_data[key] = torch.stack(value)
            else:
                val_data[key] = torch.cat(value, dim=0)
        

    if cfg.distributed and cfg.eval_ddp:
        for key, val in output.items():
            val_data[key] = sync_across_gpus(val_data[key], cfg.world_size)

    
    if cfg.local_rank == 0:
        if cfg.save_val_data:
            if cfg.distributed:
                for k, v in val_data.items():
                    val_data[k] = v[: len(val_dataloader.dataset)]
            torch.save(val_data, f"{cfg.output_dir}/fold{cfg.fold}/{pre}_data_seed{cfg.seed}.pth")
    return val_data

def cos_similarity_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def get_topk_cossim(test_emb, tr_emb, batchsize = 64, k=10, device='cuda:0',verbose=True):
    tr_emb = tr_emb.clone().float().to(torch.device(device))
    test_emb = test_emb.clone().float().to(torch.device(device))
    vals = []
    inds = []
    for test_batch in test_emb.split(batchsize):
        sim_mat = cos_similarity_matrix(test_batch, tr_emb)
        vals_batch, inds_batch = torch.topk(sim_mat, k=k, dim=1)
        vals += [vals_batch.detach()]
        inds += [inds_batch.detach()]
    vals = torch.cat(vals)
    inds = torch.cat(inds)
    return vals, inds

def run_eval(model, val_dataloader, val_index_dataloader, cfg, pre="val"):
    
    if cfg.local_rank == 0: 
        print('running eval')
    val_data = get_preds(model, val_dataloader, cfg, pre='val')
    val_index_data = get_preds(model, val_index_dataloader, cfg, pre='val_index')

#     if cfg.distributed and cfg.eval_ddp:
#         for key, val in val_data.items():
#             val_data[key] = sync_across_gpus(val_data[key], cfg.world_size)
            
    if cfg.local_rank == 0: 
        vals, inds = get_topk_cossim(val_data["embeddings"], val_index_data["embeddings"], k=1, device=cfg.device)

    #     if cfg.distributed and cfg.eval_ddp:
    #         vals = sync_across_gpus(vals, cfg.world_size)
    #         inds = sync_across_gpus(inds, cfg.world_size)

        vals = vals.data.cpu().numpy().reshape(-1)
        inds = inds.data.cpu().numpy().reshape(-1)
        labels = val_index_data["target"][inds]

        score = comp_metric(val_data["target"].cpu().numpy(), [labels, vals], n_classes=cfg.n_classes)    
  
        print('score',score)
    #             print(f"Mean {pre}_{k}", loss)
        cfg.neptune_run[f"{pre}/GAP"].log(score, step=cfg.curr_step)    


#     loss_names = [key for key in output if 'loss' in key]
#     for k in loss_names:
#         if cfg.local_rank == 0 and k in val_data:
#             losses = val_data[k].cpu().numpy()
#             loss = np.mean(losses)

#             print(f"Mean {pre}_{k}", loss)
#             cfg.neptune_run[f"{pre}/{k}"].log(loss, step=cfg.curr_step)


#     if (cfg.local_rank == 0) and (cfg.calc_metric):

#         calc_metric(cfg, val_data, pre)


    if cfg.distributed:
        torch.distributed.barrier()

    if cfg.local_rank == 0: 
        print("EVAL FINISHED")

    return 0


def run_predict_for_submission(model, test_dataloader, test_df, cfg, pre="test"):

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    test_data = defaultdict(list)

    for data in tqdm(test_dataloader, disable=cfg.local_rank != 0):

        batch = batch_to_device(data, device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        for key, test in output.items():
            test_data[key] += [output[key]]

    for key, val in output.items():
        value = test_data[key]
        if isinstance(value[0], list):
            test_data[key] = [item for sublist in value for item in sublist]
        
        else:
            if len(value[0].shape) == 0:
                test_data[key] = torch.stack(value)
            else:
                test_data[key] = torch.cat(value, dim=0)

    if cfg.distributed and cfg.eval_ddp:
        for key, test in output.items():
            test_data[key] = sync_across_gpus(test_data[key], cfg.world_size)

    if cfg.local_rank == 0:
        if cfg.save_val_data:
            if cfg.distributed:
                for k, v in test_data.items():
                    test_data[k] = v[: len(test_dataloader.dataset)]
            torch.save(test_data, f"{cfg.output_dir}/fold{cfg.fold}/{pre}_data_seed{cfg.seed}.pth")

    if cfg.local_rank == 0 and cfg.create_submission:
        
        create_submission(cfg, test_data, test_dataloader.dataset, pre)
        
    if cfg.distributed:
        torch.distributed.barrier()

    print("TEST FINISHED")


if __name__ == "__main__":

    # set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)

    cfg.distributed = False
    if "WORLD_SIZE" in os.environ:
        cfg.distributed = int(os.environ["WORLD_SIZE"]) > 1

    if cfg.distributed:

        cfg.local_rank = int(os.environ["LOCAL_RANK"])

        print("RANK", cfg.local_rank)

        device = "cuda:%d" % cfg.local_rank
        cfg.device = device
        print("device", device)
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cfg.world_size = torch.distributed.get_world_size()
        cfg.rank = torch.distributed.get_rank()
        print("Training in distributed mode with multiple processes, 1 GPU per process.")
        print(f"Process {cfg.rank}, total {cfg.world_size}, local rank {cfg.local_rank}.")
        cfg.group = torch.distributed.new_group(np.arange(cfg.world_size))
        print("Group", cfg.group)

        # syncing the random seed
        cfg.seed = int(
            sync_across_gpus(torch.Tensor([cfg.seed]).to(device), cfg.world_size)
            .detach()
            .cpu()
            .numpy()[0]
        )  #
        print("seed", cfg.local_rank, cfg.seed)

    else:
        cfg.local_rank = 0
        cfg.world_size = 1
        cfg.rank = 0  # global rank

        device = "cuda:%d" % cfg.gpu
        cfg.device = device

    set_seed(cfg.seed)

    if cfg.local_rank == 0:
        cfg.neptune_run = setup_neptune(cfg)

    train_df, val_df, test_df = get_data(cfg)
    
    train_dataset = get_train_dataset(train_df, cfg)
    train_dataloader = get_train_dataloader(train_dataset, cfg)
    
    
    val_dataset = get_val_dataset(val_df, cfg)


    
    val_dataloader = get_val_dataloader(val_dataset, cfg)
    
    val_index_dataset = get_train_dataset(train_df, cfg, allowed_targets=val_dataset.df.target.unique())
    val_index_dataloader = get_val_dataloader(val_index_dataset, cfg)
    
    if cfg.test:
        test_dataset = get_test_dataset(test_df, cfg)
        test_dataloader = get_test_dataloader(test_dataset, cfg)

    if cfg.train_val:
        train_val_dataset = get_val_dataset(train_df, cfg)
        train_val_dataloader = get_val_dataloader(train_val_dataset, cfg)

    model = get_model(cfg, train_dataset)
    model.to(device)

    if cfg.distributed:

        if cfg.syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = NativeDDP(
            model, device_ids=[cfg.local_rank], find_unused_parameters=cfg.find_unused_parameters
        )

    total_steps = len(train_dataset)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()
    
    
    
    
    
    
    for epoch in range(cfg.epochs):

        
        set_seed(cfg.seed + epoch + cfg.local_rank)

        cfg.curr_epoch = epoch
        if cfg.local_rank == 0: 
            print("EPOCH:", epoch)
        if cfg.reload_train_loader:
            train_dataset = get_train_dataset(train_df, cfg)
            train_dataloader = get_train_dataloader(train_dataset, cfg)
        
        if cfg.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1

                cfg.curr_step += cfg.batch_size * cfg.world_size

                try:
                    data = next(tr_it)
                except Exception as e:
                    print(e)
                    print("DATA FETCH ERROR")
                    # continue

                if (i == 1) & cfg.save_first_batch:
                    save_first_batch(data, cfg)

                model.train()
                torch.set_grad_enabled(True)

                # Forward pass

                batch = batch_to_device(data, device)

                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict["loss"]

                losses.append(loss.item())

                # Backward pass

                if cfg.mixed_precision:
                    scaler.scale(loss).backward()
                    if cfg.clip_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    if i % cfg.grad_accumulation == 0:

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if cfg.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    if i % cfg.grad_accumulation == 0:

                        optimizer.step()
                        optimizer.zero_grad()

                if cfg.distributed:
                    torch.cuda.synchronize()

                if scheduler is not None:
                    scheduler.step()

                if cfg.local_rank == 0 and cfg.curr_step % cfg.batch_size == 0:


#                     cfg.neptune_run["train/loss"].log(value=losses[-1], step=cfg.curr_step)
#                     cfg.neptune_run["train/box_loss"].log(value=output_dict["box_loss"].item(), step=cfg.curr_step)
#                     cfg.neptune_run["train/class_loss"].log(value=output_dict["class_loss"].item(), step=cfg.curr_step)
#                     cfg.neptune_run["train/class_loss2"].log(value=output_dict["class_loss2"].item(), step=cfg.curr_step)

                    loss_names = [key for key in output_dict if 'loss' in key]
                    for l in loss_names:
                        cfg.neptune_run[f"train/{l}"].log(value=output_dict[l].item(), step=cfg.curr_step)
                        
                    cfg.neptune_run["lr"].log(
                        value=optimizer.param_groups[0]["lr"], step=cfg.curr_step
                    )

                    progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

        if cfg.distributed:
            torch.cuda.synchronize()

        if cfg.val:
            if cfg.train_val == True:
                if (epoch + 1) % cfg.eval_train_epochs == 0 or (epoch + 1) == cfg.epochs:
                    if cfg.distributed and cfg.eval_ddp:
                        train_val_loss = run_eval(
                            model, train_val_dataloader, train_df, cfg, pre="tr"
                        )
                    else:
                        if cfg.local_rank == 0:
                            train_val_loss = run_eval(
                                model, train_val_dataloader, train_df, cfg, pre="tr"
                            )

            if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
                if cfg.distributed and cfg.eval_ddp:
                    # torch.cuda.synchronize()
                    val_loss = run_eval(model, val_dataloader, val_index_dataloader, cfg, pre="val")
                else:
                    if cfg.local_rank == 0:
#                         save_preds(model, val_dataloader, cfg, pre='val')
#                         save_preds(model, val_index_dataloader, cfg, pre='val_index')
                        val_loss = run_eval(model, val_dataloader, val_index_dataloader, cfg, pre="val")
#                         val_loss = run_eval(model, val_dataloader, val_df, cfg)
            else:
                val_score = 0

        # if cfg.local_rank == 0:

        #     if val_loss < best_val_loss:
        #         print(f'SAVING CHECKPOINT: val_loss {best_val_loss:.5} -> {val_loss:.5}')
        #         if cfg.local_rank == 0:

        #             checkpoint = create_checkpoint(model,
        #                                         optimizer,
        #                                         epoch,
        #                                         scheduler=scheduler,
        #                                         scaler=scaler)

        #             torch.save(checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_best_seed{cfg.seed}.pth")
        #         best_val_loss = val_loss

        if cfg.distributed:
            torch.distributed.barrier()

        if cfg.local_rank == 0 and cfg.epochs > 0:
            if not cfg.save_only_last_ckpt:
                checkpoint = create_checkpoint(cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler)

                torch.save(
                    checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth"
                )

    if cfg.local_rank == 0 and cfg.epochs > 0:
        # print(f'SAVING LAST EPOCH: val_loss {val_loss:.5}')
        checkpoint = create_checkpoint(cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler)

        torch.save(
            checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth"
        )

    if cfg.test:
        run_predict_for_submission(model, test_dataloader, test_df, cfg, pre="test")
#         upload_s3(cfg)
