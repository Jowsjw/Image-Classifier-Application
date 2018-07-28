# PROGRAMMER:Jinwei Wang
# DATE CREATED:7/21/2018

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import json
import PIL
from PIL import Image

def main():
    in_arg = get_input_args()
    model = loadmodel(in_arg.modelpath)
    
    device=['cuda','cpu']
    if gpu = True:
        model.to(device[0])
    else:
        print（'Please turn on the GPU mode')
        
    probs, classes = predict(in_arg.image, model, in_arg.top_k,in_arg.cat_to_name)
    probability,labels = get_flower_name(predict(in_arg.image, model, in_arg.top_k,in_arg.cat_to_name))
    
    print("image to predict:", in_arg.image)
    print("\n flower:",labels)
    print("\n probability:",probability)
    

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', type=str, help='image path')
    parser.add_argument('--modelpath', type=str, help='the path that we save our model')
    parser.add_argument('--top_k', type=int, default=5, help='return top K classes')
    parser.add_argument('--cat_to_name', type=str, default='', help='helps us to find the flower names')

    return parser.parse_args()    


def loadmodel(modelpath):
    
    checkpoint = torch.load(modelpath)
    model = checkpoint['model']
    model.state_dict=checkpoint['state_dict']
    model.classifier=checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    ig = Image.open(image)
    
    ig = ig.resize((256,256))
    ig = ig.crop((16,16,240,240))
    
    ig = np.array(ig)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    ig = (ig - mean) / std
    ig = ig.transpose(2,0,1)
    
    return ig

def predict(image, model, top_k,cat_to_name):

    image = process_image(image)
    image = torch.from_numpy(np.array([image])).float()
    model.eval()

    image = image.cuda()
    output = model.forward(image)
    ps = torch.exp(output).topk(topk)
    
    prob = ps[0][0]
    index = ps[1][0]
    
    convert = []
    for i in range(len(model.class_to_idx.items())):
        convert.append(list(model.class_to_idx.items())[i][0])

    label = []
    for i in range(topk):
        label.append(convert[index[i]])
    prob = prob.cpu()
    prob = prob.detach().numpy()
        
    return prob,label

def get_flower_name(image, model, top_k,cat_to_name):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    prob,classes = predict(image, model, top_k,cat_to_name)
    
    labels = []
    for i in classes:
        labels.append(cat_to_name[i])

    probability = []
    for i in proba:
        probability.append(i)
    return probability,labels

if __name__ == '__main__':
    main()
    
    
