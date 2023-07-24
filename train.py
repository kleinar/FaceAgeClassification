import os
import torch
import tqdm
import torch.optim as optim
import yaml
import math

from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from utils.model import AgeClassificator
from utils.datasets import train_val_test_dataloader
from utils.general import  calculate_metric

def main():
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    all_images = os.listdir(config['path-to-dataset'])
    train, test = train_test_split(all_images, train_size=0.7)
    test, val = train_test_split(test, train_size=0.5)

    model = AgeClassificator(model_name=config['model'], pretrained=config['weights'])
    device = torch.device(config['device'])
    model = model.to(device)
    criterion = torch.nn.L1Loss()
    metric = torch.nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = config['epochs']
    patience = config['patience']
    batch_size = config['batch-size']
    train_loader, val_loader, test_loader = train_val_test_dataloader(dataset_path=config['path-to-dataset'], img_size=config['img-size'], batch_size=batch_size,
                                                                                                    train_list=train, val_list= val, test_list= test)

    writer = SummaryWriter()

    # Train model
    best_metric_value = math.inf
    counter = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, data in tqdm.tqdm(enumerate(train_loader), total=int(len(train) / batch_size)):
            inputs = Variable(data['image']).to(device)
            labels = Variable(data['age']).to(device)
            optimizer.zero_grad()
            output = model(inputs).view(len(inputs))
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            metric_value = metric(output, labels)

            # write loss function result in TensorBoard
            writer.add_scalar('L1Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        model.eval()

        metric_value = calculate_metric(model, val_loader, device, metric)
        writer.add_scalar('L1Loss/val', metric_value/len(inputs), epoch)

        # Save best model
        if metric_value < best_metric_value:
            best_metric_value = metric_value
            torch.save(model.state_dict(), config['save-models-path'] + 'best_model.pt')
            counter = 0
        else:
            counter += 1

        # Early stop check
        if counter >= patience:
            print("Early stopping - model performance hasn't improved for {} epochs".format(patience))
            break

    metric_value = calculate_metric(model, test_loader, device, metric)
    print("L1Loss for test dataset: ", metric_value)
    writer.close()


if __name__ == '__main__':
    main()