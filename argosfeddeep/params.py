import os
import time
import json

data_path = r'/mnt/data'

def set_params():
    params = {
    "image_shape": [512, 512, 1],
    "patch_shape": [512, 512, 3],
    "number_of_augmentations": 2,
    "min_bound": -800,
    "max_bound": 200,
    "num_classes": 2,
    "batch_size": 4,
    "num_steps": 10000,
    "train_eval_step": 100,
    "val_eval_step": 100,
    "save_model_step": 5000,
    "learning_rate": 0.0001,
    "decay_steps": 500000,
    "decay_rate": 0.1,
    "opt_momentum": 0.9,
    "dropout_rate": 0.0,
    "l2_loss": 0.0001
    }

    if os.path.exists(os.path.join(data_path,'assets','params.json')):
        os.remove(os.path.join(data_path,'assets','params.json'))

    if not os.path.exists(os.path.join(data_path,'assets')):
        os.mkdir(os.path.join(data_path,'assets'))
    param_dir = os.path.join(data_path,'assets')
    params_file ='params.json'
    param_path = os.path.join(param_dir,params_file)
    with open(param_path,"w") as outfile:
        json.dump(params,outfile)
    time.sleep(5)