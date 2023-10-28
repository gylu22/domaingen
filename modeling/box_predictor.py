from typing import Dict, List, Optional, Tuple
import torch

from detectron2.layers import cat, cross_entropy
from detectron2.modeling.roi_heads.fast_rcnn import  FastRCNNOutputLayers
from .clip import ClipPredictor ,ClipPredictorStyle

class ClipFastRCNNOutputLayers(FastRCNNOutputLayers):

    def __init__(self,cfg, input_shape, clsnames) -> None:
        super().__init__(cfg, input_shape)
        self.cls_score = ClipPredictor(cfg.MODEL.CLIP_IMAGE_ENCODER_NAME, input_shape.channels, cfg.MODEL.DEVICE,clsnames)
        
    
    def forward(self,x,gfeat=None):
        # if x.dim() > 2:
        #     x = torch.flatten(x, start_dim=1)
        
        ## for features from clip model 
       
        if self.training:
            scores_class = self.cls_score(x[0],gfeat)
            proposal_deltas = self.bbox_pred(x[1])
            return scores_class, proposal_deltas
        else:
            scores = self.cls_score(x[0],gfeat)
            proposal_deltas = self.bbox_pred(x[1]) #self.bbox_pred(self.proj(x[0]/x[0].norm(dim=-1,keepdim=True)))
            
            return scores, proposal_deltas
        
          



class ClipFastRCNNOutputLayersStyle(FastRCNNOutputLayers):

    def __init__(self,cfg, input_shape, clsnames) -> None:
        super().__init__(cfg, input_shape)
        self.cls_score = ClipPredictorStyle(cfg.MODEL.CLIP_IMAGE_ENCODER_NAME, input_shape.channels, cfg.MODEL.DEVICE,clsnames)
        
    
    def forward(self,
                x,
                num_boxes=None,
                style_id=None):
        
        
        if self.training:
            scores_class = self.cls_score(x[0],num_boxes,style_id)
            proposal_deltas = self.bbox_pred(x[1])
            return scores_class, proposal_deltas
        else:
            # inference return the domian prompt id
            scores = self.cls_score(x[0])
            return scores
