import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy
import os
import requests
import time
import pandas
pandas.set_option('display.max_colwidth', 0)

from IPython import display
import matplotlib.pyplot as plt
from io import BytesIO
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

from PIL import Image

# Constantes / Constants
IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
STYLE_IMAGE = 'style_image'
CONTENT_IMAGE = 'content_image'
INITIAL_IMAGE = 'initial_image'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Image contenant le style à extraire
style_image_file_id = 'images/pomp1/0_0.png'
style_image = Image.open(style_image_file_id)

#image bruit blanc
content_image_file_id = 'images/whitenoise.jpg'
initial_image = Image.open(content_image_file_id)

# Image sur laquelle appliquer le style
content_image_file_id = 'images/pomp1/0_180.png'
content_image = Image.open(content_image_file_id)

images = {STYLE_IMAGE:style_image,
          CONTENT_IMAGE:content_image,
          INITIAL_IMAGE:initial_image}


#Fonctions utiles
class AddDimension(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):

        new_x = x.unsqueeze(self.dim)
        return new_x

class RemoveDimension(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):

        new_x = x.squeeze(self.dim)
        return new_x


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):

        for t, m, s in zip(x, self.mean, self.std):
            t.mul_(s).add_(m)
        new_x=x

        return new_x
        # ******

class Clamp(object):
    def __init__(self, min, max):
        self.min = float(min)
        self.max = float(max)

    def __call__(self, x):

        new_x = torch.clamp(x, min=self.min, max=self.max)
        return new_x


class Permute(object):
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, x):

        new_x=torch.permute(x, (self.dims[0], self.dims[1],self.dims[2]))
        return new_x
    

#Change tranparent background to white background
#If need
def ChangeBackground_to_white(image):
    new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
    new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
    image2 = new_image.convert('RGB') 
    

    return image2 

# Matrice de Gram
def gram_matrix(tensor):

    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())

    return gram

#Extraction des caractéristiques
def extract_features(image, model_features, layers=None):

    if layers is None:
        layers = {'0': 'conv1_1',
                  '2': 'conv1_2',
                  '5': 'conv2_1',
                  '7': 'conv2_2',
                  '10': 'conv3_1',
                  '12': 'conv3_2',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1',
                  '30': 'conv5_2'}

    features = {}
    x = image
    for layer_idx, layer in enumerate(model_features):
        x = layer(x)
        if str(layer_idx) in layers:
            features[layers[str(layer_idx)]] = x

    return features

#Perte de contenu
def func_content_loss(layer_name):
    

    content_loss = torch.mean((target_features[layer_name] - content_features[layer_name])**2)

    return content_loss

#Perte de style
def func_style_loss(weight_layer, target_gram, style_gram, target_feature):

    b, c, h, w = target_feature.shape
    c=3
    loss_layer_style = weight_layer * torch.mean((target_gram - style_gram)**2)
    style_loss = loss_layer_style / ((2* c * h * w)**2)
    return style_loss

#Perte totale
def func_total_loss(content_weight, content_loss, style_weight, style_loss):
   
    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss



#prétraitement
preprocessing = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=IMAGENET_MEAN,
                                                         std=IMAGENET_STD),
                                    AddDimension(0),
                                   ])

#post-traitement
postprocessing = transforms.Compose([RemoveDimension(0),
                                     DeNormalize(mean=IMAGENET_MEAN,
                                                 std=IMAGENET_STD),
                                     Permute([1, 2, 0]),
                                     Clamp(min=0, max=1),
                                    ])



#modèle VGG19
vgg = torchvision.models.vgg19(weights='IMAGENET1K_V1',progress=False).features

#Remplacemnet des Maxpooling par des AveragePooling
for i, layer in vgg.named_children():
    if isinstance(layer, torch.nn.MaxPool2d):
        vgg[int(i)] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

# Geler les couches pré-entraînées
for param in vgg.parameters():
    param.requires_grad = False


# Si GPU disponible, monter le modèle sur le GPU
vgg.to(DEVICE)

#Appliquer le prétraitement
pre_images = {}
for k,v in images.items():
    pre_images[k] = preprocessing(v)
    pre_images[k] = pre_images[k].to(DEVICE)

post_images = {}

#Appliquer le post-traitement
for name,img in pre_images.items():
    image = img.cpu().detach()
    post_images[name] = postprocessing(image)

#Extraction des caractéristiques
style_features=extract_features(pre_images[STYLE_IMAGE], vgg, layers=None)
content_features=extract_features(pre_images[CONTENT_IMAGE], vgg, layers=None)


#la matrice de Gram pour chaque couche de style
style_grams = {}
for layer in style_features:
    style_grams[layer] = gram_matrix(style_features[layer])

#L'image générée à partir de l'image de contenu
target=pre_images[CONTENT_IMAGE].clone().requires_grad_(True)


# Poids appliqués pour chaque couche de style
style_layers_weights = {'conv1_1': 1/5,
                        'conv2_1': 1/5,
                        'conv3_1': 1/5,
                        'conv4_1': 1/5,
                        'conv5_1': 1/5}


#Les poids pour la perte totale
content_weight = 1e-1
style_weight = 1e3


#Nombre d'itérations
steps = 2000
looks_every = 500

# Initialisation de l'optimiseur Adam. Puisqu'on modifie la cible,
optimizer = optim.Adam([target], lr=0.03)
t=0
for s in tqdm(range(1, steps+1)):

    optimizer.zero_grad()
    target_features = extract_features(target, vgg)
    layer_name = "conv4_2"
    content_loss = func_content_loss(layer_name)
    style_loss = 0 
    for l_name, l_weight in style_layers_weights.items():

        target_feature = target_features[l_name]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[l_name]
        layer_style_loss = func_style_loss(l_weight,
                                                target_gram,
                                                style_gram,
                                                target_feature)

        style_loss += layer_style_loss

    total_loss = func_total_loss(content_weight,
                                      content_loss,
                                      style_weight,
                                      style_loss)

    total_loss.backward()
    optimizer.step()

    # Afficher les images intermédiaires
    if  s % looks_every == 0:
        t+=1

        plt.figure(figsize=(10,10))
        img = target.cpu().detach()
        img_post = postprocessing(img)
        plt.imshow(img_post)
        plt.axis('off')
        #plt.imsave('/content/sample_data/FinalImage.jpg',img_post.numpy())
        plt.show()


torch.cuda.empty_cache()