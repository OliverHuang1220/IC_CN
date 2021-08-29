import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import resnet

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0)
        
        return fc, att


if __name__=='__main__':
    #model = torchvision.models.resnet50()
    net = getattr(resnet, 'resnet101')()
    model = myResnet(net)
    print(model)

    input = torch.randn(3, 224, 224) #input[batch,channel,h,w]
    fc,att= model(input)
    print(fc.shape)