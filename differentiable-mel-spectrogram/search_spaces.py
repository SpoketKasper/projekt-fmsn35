from ray import tune
from utils import get_best_device
import torch

def esc50(max_epochs):
    resample_rate = 8000
    search_space = {
        # model
        'model_name' : 'panns_cnn6',
        'n_mels' : 64,
        'hop_length' :int(resample_rate * 0.010),
        'energy_normalize' : True,
        'optimized' : True,
        'normalize_window' : False, 
        'augment' : False,

        # training
        'pretrained' : False,
	'checkpoint_path' : '/home/john/gits/differentiable-time-frequency-transforms/weights/Cnn6_mAP=0.343.pth',
        'optimizer_name' : 'adam',
        'lr_model' : 1e-4,
        'lr_tf' : 1.0, 
        'batch_size' : 32,
        'trainable' : tune.grid_search([True, False]),
        'max_epochs' : max_epochs,
        'patience' : 100,
        'device' : 'cuda:0',
        
        # dataset
        'resample_rate' : resample_rate,
        'init_lambd' : tune.grid_search([(resample_rate*x)/6 for x in [0.01, 0.035, 0.3]]),
        'dataset_name' : 'esc50', 
        'n_points' : resample_rate * 5, # hard coded zero-padding
    }

    return search_space

def audio_mnist(max_epochs):
    resample_rate = 8000
    search_space = {
        # model
        'model_name' : 'mel_linear_net',
        'n_mels' : 64,
        'hop_length' :int(resample_rate * 0.010),
        'energy_normalize' : True,
        'optimized' : True,
        'normalize_window' : False,
        'augment' : False,

        # training
        'pretrained' : False,
	'checkpoint_path' : '/home/john/gits/differentiable-time-frequency-transforms/weights/Cnn6_mAP=0.343.pth',
        'optimizer_name' : 'adam',
        'lr_model' : 1e-4,
        'lr_tf' : 1.0,
        'batch_size' : 64,
        'trainable' : tune.grid_search([True, False]),
        'max_epochs' : max_epochs,
        'patience' : 100,
        'device' : 'cuda:0',
        
        # dataset
        'resample_rate' : resample_rate,
        'init_lambd' : tune.grid_search([(resample_rate*x)/6 for x in [0.01, 0.035, 0.3]]),
        'dataset_name' : 'audio_mnist', 
        'n_points' : 8000, # hard coded zero-padding
        #'speaker_id' : tune.grid_search([[28, 56, 7, 19, 35]]),
    }

    return search_space

def time_frequency(max_epochs):
    sigma_ref = 6.38

    support = 10
    T = int(1 + (128 - (support - 1) - 1) // 8)
    init_win_length1 = torch.full((1, T), support/2)
    support = 16
    T = int(1 + (128 - (support - 1) - 1) // 8)
    init_win_length2 = torch.full((1, T), support/2)

    search_space = {
        # model
        'model_name' : 'linear_adaptive_net',
        'hop_length' : 1,
        'optimized'  : False,
        'normalize_window' : False,

        # training
        'optimizer_name' : 'sgd',
        'lr_model'       : 1e-3, 
        'lr_tf'          : 1,
        'batch_size'     : 32,
        #'trainable'      : tune.grid_search([True, False]),
        'trainable'      : True,
        'max_epochs'     : max_epochs,
        'patience'       : 100,
        'device'         : get_best_device(),
        
        # dataset
        'n_points'      : 128,
        'noise_std'     : 0.5,
        'init_lambd'    : tune.grid_search([init_win_length1, init_win_length2]),
        'n_samples'     : 5000, 
        'sigma_ref'     : sigma_ref,
        'dataset_name'  : 'time_frequency', 
        'center_offset' : False, 
    }


    return search_space
