import os
import time

data_path = r'/mnt/data'

def set_params():
    params = {
    "image_shape": [512, 512, 1],
    "patch_shape": [512, 512, 3],
    "number_of_augmentations": 2,
    "min_bound": -300,
    "max_bound": 200,
    "num_classes": 2,
    "batch_size": 4,
    "num_steps": 10,
    "train_eval_step": 5,
    "val_eval_step": 5,
    "save_model_step": 5,
    "learning_rate": 0.0001,
    "decay_steps": 500000,
    "decay_rate": 0.1,
    "opt_momentum": 0.9,
    "dropout_rate": 0.0,
    "l2_loss": 0.0001
    }

    if not os.path.exists(os.path.join(data_path,'assets')):
        os.mkdir(os.path.join(data_path,'assets'))
    param_dir = os.path.join(data_path,'assets')
    params_file ='params.json'
    param_path = os.path.join(param_dir,params_file)
    time.sleep(5)