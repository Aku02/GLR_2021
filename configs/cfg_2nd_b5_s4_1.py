from default_config import basic_cfg
import albumentations as A
import os
import pandas as pd
import cv2

cfg = basic_cfg
cfg.debug = True

# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.data_dir = "/raid/landmark-recognition-2019/"

cfg.data_folder = cfg.data_dir + "train/"
cfg.train_df = "/mount/glr2021/data/2021/train.csv"

cfg.val_df = '/raid/landmark-recognition-2019/' + "recognition_solution_v2.1.csv"
cfg.output_dir = f"/mount/glr2021/models/{os.path.basename(__file__).split('.')[0]}"
cfg.val_data_folder = "/raid/landmark-recognition-2019/" + "test/"

cfg.test = False
cfg.test_data_folder = cfg.data_dir + "test/"
# cfg.test_df = cfg.data_dir + "sample_submission_v1.csv"

cfg.eval_retrieval = True
cfg.query_data_folder = "/raid/landmark-recognition-2019/" + "test/"
cfg.index_data_folder = "/raid/landmark-recognition-2019/" + "index/"
cfg.query_df = '/mount/glr2021/data/2019/query_v2.csv'
cfg.index_df = '/mount/glr2021/data/2019/index_v2.csv'

#logging
cfg.neptune_project = "christofhenkel/glr2021"
cfg.neptune_connection_mode = "debug"
cfg.tags = "debug"





# MODEL
cfg.model = "ch_mdl_2nd_1"
cfg.backbone = "efficientnet_b5"
cfg.neck = "option-D"
cfg.embedding_size = 512
cfg.pool = "gem"
cfg.gem_p_trainable = True
cfg.pretrained = False
cfg.pretrained_weights = '/mount/glr2021/models/glr2020-2nd-place/models/v2x_sgd_ls_aug3b_norm1_0918_class_efficientnet_b5_gem_fc_arcface2_1head_i640/1.60_headless.pth'
cfg.pretrained_weights_strict = True
cfg.headless = True

# DATASET
cfg.dataset = "ch_ds_2nd_1"
cfg.normalization = False
cfg.landmark_id2class_id = pd.read_csv('./assets/landmark_id2class.csv')
cfg.num_workers = 8
# cfg.data_sample = 100000
cfg.loss = 'adaptive_arcface'
cfg.arcface_s = 30
cfg.arcface_m = 0.3



# OPTIMIZATION & SCHEDULE

# cfg.fold = 0
cfg.lr = 0.001
# cfg.optimizer = "adam"
cfg.weight_decay = 1e-4
cfg.warmup = 1
cfg.epochs = 1
cfg.batch_size = 16
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 4.

#inference
cfg.train = False
cfg.val = True
cfg.test = False
cfg.save_val_data = True
cfg.train_val = False
cfg.save_only_last_ckpt = False
cfg.eval_ddp =True
cfg.img_size = 704
# AUGS

cfg.train_aug = None

cfg.val_aug = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)])



