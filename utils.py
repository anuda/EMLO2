import io
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms 
from PIL import Image
import torchvision.transforms.functional as trans_f



idx_to_label={0: 'airplane',
1: 'automobile',
2: 'bird',
3: 'cat',
4: 'deer',
5: 'dog',
6: 'frog',
7: 'horse',
8: 'ship',
9: 'truck',}

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
model = NeuralNet()

PATH = "cifar10_ffn.pth"
model.load_state_dict(torch.load(PATH))
model.eval()

def transform_image(image_bytes):
    

    image = Image.open(io.BytesIO(image_bytes))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((32,32))])
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    # images = trans_f.resize(1,3,32,32)
    print('here')
    print(image_tensor.shape)
    outputs = model(image_tensor)
    print(outputs)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return idx_to_label[predicted.item()]

