import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import random 
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class ClipRN101(Backbone):
    def __init__(self, cfg, clip_visual):
        super().__init__()
        self.enc = None
        self.unfreeze = cfg.MODEL.BACKBONE.UNFREEZE 
        self.proj = nn.Linear(512,512)
        self.global_proj = nn.Linear(512,512)
        self.use_proj = cfg.MODEL.USE_PROJ 
        

    def set_backbone_model(self,model):
        self.enc = model
        for name,val in self.enc.named_parameters():
            head = name.split('.')[0]
            if head not in self.unfreeze:
                val.requires_grad = False
                # print(f'{name}:{val.requires_grad}')
            else:
                val.requires_grad = True
                # print(f'{name}:{val.requires_grad}')
        
        self.backbone_unchanged = nn.Sequential(*self.enc.layer3[:19])


    def forward(self, image,mean=None,std=None):
        x = image
        x = self.enc.relu1(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu2(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu3(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)
        
        x = self.enc.layer1(x)
        if self.training is True:
            x = self.instance_norm(x,mean,std)
        x = self.enc.layer2(x)
        x = self.enc.layer3(x)
        return {"res4": x}


    def forward_l12(self, image):
        x = image
        x = self.enc.relu1(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu2(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu3(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)
        
        x = self.enc.layer1(x)
        x = self.enc.layer2(x)  
       
        return x
     
    def forward_l3(self, x):
        
        x = self.enc.layer3(x)
        return {"res4": x}
 
    def output_shape(self):
        return {"res4": ShapeSpec(channels=1024, stride=16)}
    
    def forward_res5(self,x):
        # detectron used last resnet layer for roi heads
        return self.enc.layer4(x)

    def attention_global_pool(self,input):
        x = input
        x = self.enc.attnpool(x)
        return x


    def instance_norm(self,
                        x:torch.Tensor,
                        beta:torch.Tensor, 
                        gamma:torch.Tensor,
                        p = 0.5) -> torch.Tensor:
        
        if random.random() > p:
            return x
        B = x.size(0)
        eps = 1e-6
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig
        return x_normed * gamma.view(gamma.shape[0],gamma.shape[1],1,1) + beta.view(beta.shape[0],beta.shape[1],1,1)
    
    
    
    
    
@BACKBONE_REGISTRY.register()
class ClipRN101Denoise(Backbone):
    
    def __init__(self,cfg,clip_visual):
        super().__init__()
        self.unfreeze = cfg.MODEL.BACKBONE.UNFREEZE
    

    def set_backbone_model(self,model):
        self.enc = model
        
        
    def forward(self, image):
        x = image
        x = self.enc.relu1(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu2(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu3(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)
        
        x = self.enc.layer1(x)
        x = self.enc.layer2(x)
        return {"res3":x} 
    
    
    def forward_res4(self,x):
        x = self.enc.layer3(x)
        return {"res4":x}
   
    def forward_res5(self,x):
        #detectron used last resnet layer for roi heads
        return self.enc.layer4(x)

    def attention_global_pool(self,input):
        x = input
        x = self.enc.attnpool(x)
        return x
    
    def output_shape(self):
        return {"res4": ShapeSpec(channels=1024, stride=16)}