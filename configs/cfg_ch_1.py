from default_config import basic_cfg
import albumentations as A
import os
import pandas as pd

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

#logging
cfg.neptune_project = "christofhenkel/glr2021"
cfg.neptune_connection_mode = "debug"
cfg.tags = "debug"





# MODEL
cfg.model = "ch_mdl_1"
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.neck = "option-D"
cfg.embedding_size = 512

# DATASET
cfg.dataset = "ch_ds_1"
cfg.normalization = 'simple'
cfg.landmark_id2class_id = pd.read_csv('/mount/glr2021/landmark_id2class.csv')
cfg.num_workers = 16
# cfg.data_sample = 100000



#OPTIMIZATION & SCHEDULE

# cfg.fold = 0
cfg.lr = 0.0005
cfg.epochs = 10
cfg.batch_size = 32
cfg.mixed_precision = True
cfg.pin_memory = False

#inference
cfg.val = True
cfg.test = False
cfg.save_val_data = False
cfg.train_val = False
cfg.save_only_last_ckpt = True

#AUGS


cfg.train_aug = A.Compose(
    [
        A.Resize(256,256,p=1.),
    ],

)

cfg.val_aug = A.Compose(
    [
        A.Resize(256,256,p=1.),
    ],

)

