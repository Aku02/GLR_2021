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
cfg.data_folder = cfg.data_dir + "train/"
cfg.train_df = cfg.data_dir + "train.csv"
cfg.val_df = '/raid/landmark-recognition-2019/' + "recognition_solution_v2.1.csv"
cfg.output_dir = f"/mount/glr2021/models/{os.path.basename(__file__).split('.')[0]}"
cfg.val_data_folder = "/raid/landmark-recognition-2019/" + "test/"
cfg.test_data_folder = cfg.data_dir + "test/"
# cfg.test_df = cfg.data_dir + "sample_submission_v1.csv"

cfg.eval_retrieval = True
cfg.query_data_folder = "/raid/landmark-recognition-2019/" + "test/"
cfg.index_data_folder = "/raid/landmark-recognition-2019/" + "index/"
cfg.query_df = '/mount/glr2021/data/2019/query_v2.csv'
cfg.index_df = '/mount/glr2021/data/2019/index_v2.csv'

#logging
cfg.neptune_project = "christofhenkel/glr2021"
cfg.neptune_connection_mode = "async"
cfg.tags = "debug"





# MODEL
cfg.model = "ch_mdl_3"
cfg.backbone = "gluon_seresnext101_32x4d"
cfg.neck = "option-D"
cfg.embedding_size = 512
cfg.pool = "gem"
cfg.gem_p_trainable = True
cfg.pretrained_weights = '/mount/glr2021/models/config_ch_28_m03/ckpt_converted_headless.pth'
cfg.pretrained_weights_strict = True
cfg.headless = True
# DATASET
cfg.dataset = "ch_ds_2"
cfg.normalization = 'imagenet'
cfg.landmark_id2class_id = pd.read_csv('./assets/landmark_id2class.csv')
cfg.num_workers = 8
# cfg.data_sample = 100000



# OPTIMIZATION & SCHEDULE

# cfg.fold = 0
cfg.lr = 0.0005
cfg.epochs = 1
cfg.batch_size = 32
cfg.mixed_precision = True
cfg.pin_memory = False

#inference
cfg.train = False
cfg.val = True
cfg.test = False
cfg.save_val_data = True
cfg.train_val = False
cfg.save_only_last_ckpt = True
# cfg.eval_ddp =True
# AUGS

cfg.train_aug = None


cfg.val_aug = A.Compose([ A.SmallestMaxSize(512),
                         
                     A.CenterCrop(always_apply=False, p=1.0, height=512, width=512), 
                    ], p=1.0)




