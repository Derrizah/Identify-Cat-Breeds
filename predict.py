import torch
import numpy as np
import argparse
import json
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Configuring the parser
parser = argparse.ArgumentParser()

parser.add_argument('image_path',
                    help='Image directory')
parser.add_argument('checkpoint',
                    help='model checkpoint')
parser.add_argument('--top_k', action='store',
                    dest='topk',
                    default=5,
                    help='top k prediction probabilities')
parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    default=None,
                    help='json mapping')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='set gpu')

# Parse arguments
arguments = parser.parse_args()

checkpoint = torch.load(arguments.checkpoint)
model = models.vgg16(pretrained=True)

# We shouldn't compute gradients
for param in model.parameters():
    param.requires_grad = False

    # Creating new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 1024)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(checkpoint['layers'][0], checkpoint['layers'][1])),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.2)),
        ('fc4', nn.Linear(checkpoint['layers'][1], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']

device = ("cuda" if arguments.gpu else "cpu")
model.to(device)

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Process a PIL image for use in a PyTorch model
pil_image = Image.open(arguments.image_path)

process_image = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])


image = process_image(pil_image).to(device)
# Adding dimension to image (first dimension)
np_image = image.unsqueeze_(0)

model.eval()
with torch.no_grad():
    log_ps = model.forward(np_image)

ps = torch.exp(log_ps)
top_k, top_classes_idx = ps.topk(int(arguments.topk), dim=1)
top_k, top_classes_idx = np.array(top_k.to('cpu')[0]), np.array(top_classes_idx.to('cpu')[0])

# Inverting dictionary
idx_to_class = {x: y for y, x in model.class_to_idx.items()}

top_classes = []
for index in top_classes_idx:
    top_classes.append(idx_to_class[index])
    
if arguments.category_names != None:
    with open(arguments.category_names, 'r') as f:
        cat_to_name = json.load(f)
        top_class_names = [cat_to_name[top_class] for top_class in list(top_classes)]
        print(f'Top {arguments.topk} probabilities: {list(top_k)}')
        print(f'Top {arguments.topk} classes: {top_class_names}')
else:
    print(f'Top {arguments.topk} probabilities: {list(top_k)}')
    print(f'Top {arguments.topk} classes: {list(top_classes)}')