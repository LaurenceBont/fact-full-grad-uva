import argparse
from utils import ModelConfiguration, DataLoaderConfiguration
import os
from roar_experiment import experiment

def main(config):
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    if config.experiment == 'roar':
        model_config = ModelConfiguration(epochs=80, learning_rate=0.01, checkpoint_path=PATH + 'saved-models/')
        loader_config = DataLoaderConfiguration(path=PATH, data_dir='dataset/')
        experiment(model_config, loader_config)

    elif config.experiment == 'pixel_perturbation':
        model_config = ModelConfiguration()
        loader_config = DataLoaderConfiguration()

        percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
        experiment(model_config, loader_config, percentages)

    elif config.experiment == 'extra':
        model_config = ModelConfiguration()
        loader_config = DataLoaderConfiguration()        
    else:
        print("experiment does not exist, please select roar, pixel_perturbation or extra")

if __name__== "__main__":
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--experiment', type=str, default='roar', help="Select roar, pixel_perturbation or extra experiment")
    config = parser.parse_args()
    main(config)



        