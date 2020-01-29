import argparse
from utils import ModelConfiguration, DataLoaderConfiguration

def main(config):
    if config.experiment == 'roar':
        model_config = ModelConfiguration()
        loader_config = DataLoaderConfiguration()

    elif config.experiment == 'pixel_perturbation':
        model_config = ModelConfiguration()
        loader_config = DataLoaderConfiguration()
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
