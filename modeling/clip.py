import clip
import torch
import torch.nn as nn
import numpy as np
import copy

DOMIAN_PROMPTS = ["an image taken on a clear day",
                "an image taken on a snow night",
                "an image taken on a fog night",
                "an image taken on a cloudy night",
                "an image taken on a rain night",
                "an image taken on a heavy rain night",
                "an image taken on a stormy night",
                "an image taken on a snow day",
                "an image taken on a fog day",
                "an image taken on a cloudy day",
                "an image taken on a rain day",
                "an image taken on a heavy rain day",
                "an image taken on a stormy day",
                "an image taken on a snow evening",
                "an image taken on a fog evening",
                "an image taken on a cloudy evening",
                "an image taken on a rain evening",
                "an image taken on a heavy evening",
                "an image taken on a stormy evening"]


class ClipPredictor(nn.Module):
    def __init__(self, clip_enocder_name,inshape, device, clsnames):
        super().__init__()
        self.model, self.preprocess = clip.load(clip_enocder_name, device)
        self.model.float()
        #freeze everything
        for name, val in self.model.named_parameters():
            val.requires_grad = False
        # this is only used for inference   
        self.frozen_clip_model = copy.deepcopy(self.model)

        self.visual_enc = self.model.visual
        prompt = 'a photo of a {}'
        print(clsnames)
        with torch.no_grad():
            # class wise text 
            text_inputs = torch.cat([clip.tokenize(prompt.format(cls)) for cls in clsnames]).to(device)
            self.text_features = self.model.encode_text(text_inputs).float()    
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        # 
        self.projection = nn.Linear(inshape,512)
       
        
   

    def forward(self, feat, gfeat=None):

        if feat.shape[-1] > 512:
            feat = self.projection(feat)
        
        feat = feat/feat.norm(dim=-1,keepdim=True)
        if gfeat is not None:
            
            feat = feat-gfeat
            feat = feat/feat.norm(dim=-1,keepdim=True)
        scores =  (100.0 * torch.matmul(feat,self.text_features.detach().T))

        # print(scores.min(),scores.max())
        # add for bkg class a score 0
        scores = torch.cat([scores,torch.zeros(scores.shape[0],1,device=scores.device)],1) 
        return scores
                                            

    

class ClipPredictorStyle(nn.Module):
    
    def __init__(self, clip_enocder_name,inshape, device, clsnames):
        super().__init__()
        self.model, self.preprocess = clip.load(clip_enocder_name, device)
        self.model.float()
        #freeze everything
        for name, val in self.model.named_parameters():
            val.requires_grad = False
        
        with torch.no_grad():
            text_features_list = []
            for prompt in DOMIAN_PROMPTS:
                text_inputs = torch.cat([clip.tokenize(prompt.replace('an image', 'an image of ' + f'{cls}'))
                                        for cls in clsnames]).to(device)
                text_features = self.model.encode_text(text_inputs).float()    
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features_list.append(text_features)
        self.text_features = torch.cat(text_features_list,dim=0)        
        self.clsnames = clsnames        
        # this is only used for inference   
        self.visual_enc = self.model.visual
        self.projection = nn.Linear(inshape,512)

    
    def forward(self, feat, num_boxes=None,domian_idx=None):
        device = feat.device
        if num_boxes is not None:
              assert feat.shape[0] == torch.sum(num_boxes) 
        if feat.shape[-1] > 512:
            feat = self.projection(feat)
        # visual feature norm
        num_class = len(self.clsnames) 
        feat = feat/feat.norm(dim=-1,keepdim=True)
        # feat_start_idx = [sum(num_boxes[:i]) for i in range(len(num_boxes))]

        if self.training:
            score_list = []
            for i,idx in enumerate(domian_idx):
                if i == 0:
                    feat_ = feat[0:num_boxes[i]]
                else:
                    start = torch.sum(num_boxes[0:i])
                    end = start + num_boxes[i]
                    feat_ = feat[start:end,:]    
                score = (100.0 * torch.matmul(feat_,self.text_features[idx*num_class:(idx+1)*num_class].detach().T))
                score_list.append(score)
            scores = torch.cat(score_list,dim=0)
            scores = torch.cat([scores,torch.zeros(scores.shape[0],1,device=device)],1) 
            return scores
        else:
            domain_score_list =[]
            for j in range(len(DOMIAN_PROMPTS)):
                domain_score = (100.0 * torch.matmul(feat,self.text_features[j*num_class:(j+1)*num_class].detach().T))
                domain_score = torch.cat([domain_score,torch.zeros(domain_score.shape[0],1,device=device)],1)
                domain_score_list.append(domain_score)
            return torch.cat(domain_score_list,dim=-1)
        
                
                
        
     

    
