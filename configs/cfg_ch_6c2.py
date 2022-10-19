from default_config import basic_cfg
import albumentations as A
import os
import pandas as pd
import cv2

cfg = basic_cfg
cfg.debug = True

# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.data_dir = "/raid/landmark-recognition-2021/"

cfg.train
cfg.data_folder = cfg.data_dir + "train/"
cfg.train_df = cfg.data_dir + "train.csv"

cfg.val
cfg.val_df = '/raid/landmark-recognition-2019/' + "recognition_solution_v2.1.csv"
cfg.output_dir = f"/mount/glr2021/models/{os.path.basename(__file__).split('.')[0]}"
cfg.val_data_folder = "/raid/landmark-recognition-2019/" + "test/"

cfg.test = False
cfg.test_data_folder = cfg.data_dir + "test/"
# cfg.test_df = cfg.data_dir + "sample_submission_v1.csv"

cfg.eval_retrieval = False
cfg.query_data_folder = "/raid/landmark-recognition-2019/" + "test/"
cfg.index_data_folder = "/raid/landmark-recognition-2019/" + "index/"

#logging
cfg.neptune_project = "christofhenkel/glr2021"
cfg.neptune_connection_mode = "async"
cfg.tags = "debug"





# MODEL
cfg.model = "ch_mdl_4"
cfg.backbone = "tf_efficientnet_b3_ns"
cfg.neck = "option-D"
cfg.embedding_size = 512
cfg.pool = "gem"
cfg.gem_p_trainable = True
# cfg.pretrained_weights = '/mount/glr2021/models/config_aws_11/ckpt_converted.pth'
# cfg.pretrained_weights_strict = True
# DATASET
cfg.dataset = "ch_ds_4"
cfg.normalization = 'imagenet'
cfg.landmark_id2class_id = pd.read_csv('./assets/landmark_id2class.csv')
cfg.num_workers = 8
# cfg.data_sample = 100000
cfg.loss = 'adaptive_arcface'
cfg.arcface_s = 45
cfg.arcface_m = 0.3


# OPTIMIZATION & SCHEDULE

# cfg.fold = 0
cfg.lr = 0.0015
# cfg.optimizer = "adam"
cfg.weight_decay = 1e-4
cfg.warmup = 1
cfg.epochs = 10
cfg.batch_size = 64
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 1.

#inference
cfg.train = True
cfg.val = True
cfg.test = False
cfg.save_val_data = True
cfg.train_val = False
cfg.save_only_last_ckpt = False
cfg.eval_ddp =True

# AUGS

cfg.train_aug = A.Compose([ A.LongestMaxSize(512,p=1),
                            A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT,p=1),
                            A.RandomCrop(always_apply=False, p=1.0, height=448, width=448), 
                            A.HorizontalFlip(always_apply=False, p=0.5), 
                           # A.Lambda(name="cropout", image=cropout_border, p=0.5)
                           ],
                            p=1.0
                            )

cfg.val_aug = A.Compose([ A.LongestMaxSize(512,p=1),
                             A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT,p=1),
                            A.CenterCrop(always_apply=False, p=1.0, height=448, width=448), 
                            ], 
                            p=1.0
                            )



