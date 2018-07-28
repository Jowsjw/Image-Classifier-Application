# PROGRAMMER:Jinwei Wang
# DATE CREATED:7/21/2018

import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms,models
import jason

def main():
    in_arg = get_input_args()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    data = load_data(in_arg.data_dir)
    model = build_model(in_arg.arch,in_arg.hidden_units,in_arg.output_units)
    
    train_model(data,in_arg.epos,in_arg.gpu,in_arg.learning_rate,model)
    model_validation(data,model)
    
    save_checkpoint(model, in_arg.learning_rate, in_arg.hidden_units, in_arg.epos, in_arg.save_dir)
    
    
def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, help='image path')
    parser.add_argument('--save_dir', type=str, help='checkpoint path')
    parser.add_argument('--arch', type=str, default='vgg16', help='models for image classification')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=1000, help='hidden layers')
    parser.add_argument('--output_units', type=int, default=102, help='output layer')
    parser.add_argument('--epos', type=int, default=3, help='the number of epochs')
    parser.add_argument('--gpu', type=bool, default=False, help='train on gpu mode')

    return parser.parse_args()
    
def load_data(data_dir):f
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                         ]) 

    valid_transform = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                         ])

    test_transform = transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                         ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return {
                'train': trainloader,
                'valid': validloader,
                'test': testloader
    }

def build_model(arch,hidden_units,output_units):

    model = model.arch(pretrained=True)
    
    for param in model.parameters():
    param.requires_grad = False
    
    NewClassifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(hidden_units, output_units)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = NewClassifier
    
    return model

def train_model(data,epos,gpu,learning_rate,model):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    device=['cuda','cpu']
    epochs = epos
    print_every = 40
    steps = 0

    print('Training Start!')
    if gpu = True:
        model.to(device[0])
    else:
        print（'Please turn on the GPU mode')

    for e in range(epochs):
        running_loss = 0
        model.train()
        for i, (inputs, labels) in enumerate(data['train']):
            steps += 1

            inputs, labels = inputs.to(device[0]), labels.to(device[0])

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():

                    valid_loss = 0
                    valid_accuracy = 0

                    for valid_inputs, valid_labels in data['valid']:
                        valid_inputs, valid_labels = valid_inputs.to(device[0]), valid_labels.to(device[0])

                        valid_output = model.forward(valid_inputs)
                        valid_loss += criterion(valid_output, valid_labels).item()

                        ps = torch.exp(valid_output)
                        equality = (valid_labels.data == ps.max(dim=1)[1])
                        valid_accuracy += equality.type(torch.FloatTensor).mean()

                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Train Loss: {:.4f}...".format(running_loss),
                          "Validation Loss: {:.4f}...".format(valid_loss),
                          "Validation Accuracy: {:.4f}...".format(valid_accuracy/len(data['valid'])))

                    running_loss = 0
                    model.train()

    print('Finish Training!')

def model_validation(data,model):
    device=['cuda','cpu']
    test_loss = 0
    test_accuracy = 0

    model.eval()

    for test_inputs, test_labels in data['test']:

        with torch.no_grad():
            test_inputs, test_labels = test_inputs.to(device[0]), test_labels.to(device[0])

            test_output = model.forward(test_inputs)
            loss = criterion(test_output, test_labels)
            test_loss += loss.item()

            ps = torch.exp(test_output)
            equality = (test_labels.data == ps.max(dim=1)[1])

            test_accuracy += equality.type(torch.FloatTensor).mean()

    print("Test Loss: {:.3f}...".format(test_loss),
          "Test Accuracy : {:.3f}".format(test_accuracy/len(data['test'])))

def save_checkpoint(model, learning_rate, hidden_units, epochs, save_dir):
    model = {
        'model': model,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(model, save_dir)
    
if __name__ == "__main__":
    main()    
    