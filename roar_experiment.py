"""
    This file contains all the functions needed to run the roar experiment
"""

def experiment(criterion, optimizer, scheduler, cfg, percentages = [0.1, 0.3, 0.5, 0.7, 0.9]):
    
    # If adjusted data is not created, create it. 
    if not os.path.exists('dataset/cifar-100-adjusted'):
        get_salience_based_adjusted_data(sample_loader, ks, percentages, dataset = "train")
        get_salience_based_adjusted_data(sample_loader, ks, percentages, dataset = "test")

    # Train model based on certrain adjusted data
    accuracy_list = do_experiment(model, criterion, optimizer, scheduler, percentages, cfg)

    # Create plot
    plt.plot(percentages, accuracy_list, marker = 'o')
    plt.show()

def do_experiment(model, criterion, optimizer, scheduler, percentages, cfg, num_classes = 10):
    accuracy_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    if num_classes == 10:
        transform = [CIFAR_10_TRANSFORM, CIFAR_10_TRANSFORM]
    else:
        transform = [CIFAR_100_TRANSFORM_TRAIN, CIFAR_100_TRANSFORM_TEST]

    for percentage in percentages:
        copied_model = copy.deepcopy(model)

        data_dir = f"dataset/cifar-100-adjusted/cifar-{num_classes}-{percentage}%-removed/"
        adjusted_train_data = load_imageFolder_data(cfg.batch_size, transform[0], cfg.shuffle, cfg.num_workers, data_dir + "train")
        adjusted_test_data = load_imageFolder_data(cfg.batch_size, transform[1], cfg.shuffle, cfg.num_workers, data_dir + "test")
        
        train(copied_model, criterion, optimizer, scheduler, adjusted_train_data, adjusted_test_data, device,
        checkpoint_path, model_name, epochs, save_epochs)

        accuracy_list.append(parse_epoch(adjusted_test_data, model_k_data, None, criterion, device, train=False))

if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_name', type=str, default="VGG-11", help="Name of the model when saved")
    parser.add_argument('--num_classes', type=int, default=100, help='Dimensionality of output sequence')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs until break')
    parser.add_argument('--load_model', type=str, default='', help='Give location of weights to load model')
    parser.add_argument('--save_epochs', type=int, default=1, help="save model after epochs")
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda'")
    parser.add_argument('--save_model', type=bool, default=True, help="If set to false the model wont be saved.")
    parser.add_argument('--data_dir', type=str, default=PATH + 'dataset', help="data dir for dataloader")
    parser.add_argument('--dataset_name', type=str, default='/cifar-10-imageFolder', help= "Name of dataset contained in the data_dir")
    parser.add_argument('--checkpoint_path', type=str, default=PATH + 'saved-models/', help="model saving dir.")

    config = parser.parse_args()

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    config.checkpoint_path = os.path.join(config.checkpoint_path, '{model}-{epoch}-{type}.pth')


    device = torch.device(config.device)

    model = vgg11(pretrained=False, im_size = (3, 32, 32), num_classes=config.num_classes, class_size=512).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
    
    percentages = [0.10, 0.33, 0.66, 0.99]
    experiment(criterion, optimizer, scheduler, config, percentages)
        




