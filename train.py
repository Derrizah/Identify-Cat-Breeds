import torch
import argparse
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Configuring the parser
parser = argparse.ArgumentParser()

parser.add_argument('data_dir',
                    help='files directory')
parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='where to save model')
parser.add_argument('--arch', action='store',
                    dest='arch',
                    default='vgg16',
                    help='model architecture')
parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    default=0.003,
                    help='learning rate')
parser.add_argument('--hidden_layers', action='append',
                    dest='hidden_layers',
                    default=[1024, 512],
                    help='hidden layers count')
parser.add_argument('--ephocs', action='store',
                    dest='epochs',
                    default=5,
                    help='epoch count')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='set gpu')
# Parse arguments
arguments = parser.parse_args()

# Decide device
device = ("cuda" if arguments.gpu else "cpu")

# Declaring transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(arguments.data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(arguments.data_dir + '/valid', transform=valid_transforms)


# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)

# Trying to load the arch given by the user
try:
    model = eval("models." + arguments.arch +"(pretrained=True)")
except AttributeError:
    print("Model not available. Loading default.")
    model = models.vgg16(pretrained=True)
    

for param in model.parameters():
    param.requires_grad = False

# Defining our own classifier for the model
classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 1024)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p=0.2)),
                            ('fc2', nn.Linear(int(arguments.hidden_layers[0]), int(arguments.hidden_layers[1]))),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p=0.2)),
                            ('fc4', nn.Linear(int(arguments.hidden_layers[1]), 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)
model.to(device)

# Training the model
epochs = arguments.epochs
running_loss = 0

train_losses, valid_losses = [], []

for e in range(epochs):
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        log_ps = model.forward(inputs)
        loss = criterion(log_ps, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)                               
                log_ps = model.forward(inputs)
                batch_loss = criterion(log_ps, labels)
                
                valid_loss += batch_loss.item()
                
                # Accuracy
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {e+1}/{epochs}.. "
              f"Train loss: {running_loss/len(validloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}.. ")
        
        train_losses.append(running_loss)
        valid_losses.append(valid_loss)
        running_loss = 0
        model.train()

# Creating checkpoint data
model.class_to_idx = train_data.class_to_idx
checkpoint = {'epochs': epochs,
              'train_losses': train_losses,
              'valid_losses': valid_losses,
              'class_to_idx': model.class_to_idx,
              'layers': arguments.hidden_layers,
              'optimizer_state_dict': optimizer.state_dict(), 
              'model_state_dict': model.state_dict()}

# Saving for later use
torch.save(checkpoint, 'checkpoint.pth')