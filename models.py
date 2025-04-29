# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (27) and a Sigmoid instead of a default Softmax.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from transformers import ViTForImageClassification
from timm.models import vit_base_patch16_224_in21k, vit_large_patch16_224_in21k
from config import get_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
#from simclr import SimCLR
import os
import random
import numpy as np
import argparse
import torchvision
from torchvision import datasets, models, transforms
import config
import timm
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
config = get_config()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True,task='train'):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    
    

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet34":
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224  
            
    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224 
        
    elif model_name == "resnet101":
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224  

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg11_bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        #print(model_ft)
        
    elif model_name == "vgg16":
        
        model_ft = models.vgg16(pretrained=True)

        num_ftrs = model_ft.classifier[6].in_features
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        #print(model_ft)
        
    elif model_name == "vgg19":
        
        model_ft = models.vgg19(pretrained=True)

        num_ftrs = model_ft.classifier[6].in_features
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        #print(model_ft)
        
    elif model_name == "vgg11":
        
        model_ft = models.vgg11(pretrained=True)

        num_ftrs = model_ft.classifier[6].in_features
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        #print(model_ft)
        
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet121":
        """ Densenet
        """
        
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        #print(model_ft)
        input_size = 224

    elif model_name == "densenet169":
        """ Densenet
        """
        model_ft = models.densenet169(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "densenet201":
        """ Densenet
        """
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.aux_logits = False
        model_ft.AuxLogits = None
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        # # Handle the auxilary net
        # num_ftrs = model_ft.AuxLogits.fc.in_features
        # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # # Handle the primary net
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "mobilenet":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "vitbase":
        model_ft = vit_base_patch16_224_in21k(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        #print(model_ft)
    
    elif model_name == "vitlarge":
        model_ft = vit_large_patch16_224_in21k(pretrained=use_pretrained)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        #print(model_ft)
           
    elif model_name == "resnext":
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, out_features=num_classes)
        
    elif model_name == "ConvNext":
        model_ft = timm.create_model('convnextv2_atto.fcmae', pretrained=use_pretrained, num_classes=4)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == "simclr":
        print()
        
    elif model_name == "efficientNet":
        model_ft = models.efficientnet_b0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
        
    elif model_name == "swinlarge":
        #'swin_tiny_patch4_window7_224'
        #model_ft = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=use_pretrained)
        #model_ft = timm.create_model('swin_large_patch4_window7_224.ms_in22k', pretrained=use_pretrained)
        # get model specific transforms (normalization, resize)
        # data_config = timm.data.resolve_model_data_config(model_ft)
        # transforms = timm.data.create_transform(**data_config, is_training=False)
        # outputs = model.forward_features(transforms(images).unsqueeze(0))
        # outputs = model.forward_head(outputs, pre_logits=True)
        
        HUB_URL = "SharanSMenon/swin-transformer-hub:main"
        MODEL_NAME = "swin_tiny_patch4_window7_224"
        # check hubconf for more models.
        model_ft = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=use_pretrained) # load from torch hub
        num_ftrs = model_ft.head.in_features
        #print(num_ftrs)
        
        model_ft.head = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()
        
    if task == 'train':
        print("training the model")
        return model_ft,input_size
    
    else:
        print("testing the model")
        
        # #path = "cxr14_" + model_name + '_' + task + '.pth'   #change cxr14_ to chexp for chexpert
        # path = "./saved_models/"  + model_name + 'nfold' '.pth'
        
        # print(path)   
        # # make the change
        
        # model_ft.load_state_dict(torch.load(path))
        return model_ft, input_size



#Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# models_choices = ['resnet18','resnet34', 'resnet50', 'alexnet', 'vgg11_bn','vgg16', 'vgg19', 'squeezenet', 
#                   'densenet121', 'densenet169', 'densenet201'  'inception', 'diet']
# model_name = "swinlarge"

# #Number of classes in the dataset
# num_classes = 4
# # # Flag for feature extracting. When False, we finetune the whole model,
# # #   when True we only update the reshaped layer params
# feature_extract = False
# # # Initialize the model for this run
# model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, task ='train')
# #print(model)

# dummy_x = torch.randn(1, 3, 224, 224)
# logits = model(dummy_x)  # (1,3)
# print(model)
# print(logits.shape)




















