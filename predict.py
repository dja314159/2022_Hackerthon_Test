from flask import request
from flask_restx import Namespace, Resource
import base64
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch): 
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

        
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]
  



############## 위 부분은 모델을 불러오는데 필요한 클래스 및  함수 불러오는 부분 ################


predict_ns = Namespace('Predict', path='/api/v1', description='predict')

swagger_parser = predict_ns.parser()


model = ResNet()

device = get_default_device()

#모델 경로는 알아서 바꾸기
loaded_model = torch.load('model.pt') 



@predict_ns.route('/predict')
class Predict(Resource):
    """ POST로 받은 이미지를 분류하는 API """

    @predict_ns.doc(responses={
        200: 'OK',
        204: 'Fail to find a face',
        400: 'Bad Request',
        500: 'Internal Server Error'
    })
    @predict_ns.expect(swagger_parser)
    def post(self):
        if "image" not in request.files:
            raise ValueError
        
        input_image = request.files["image"].read()

        predicted = model.predict_image(input_image, loaded_model)
        # 예측 결과로 리턴하는 label 종류
        # ['metal', 'cardboard', 'paper', 'plastic', 'trash', 'glass']
        
        response = {
            "predict": predicted
        }
        print(predicted)

        return response, 200