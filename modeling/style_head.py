import torch
from typing import Dict, List, Optional, Tuple
from detectron2.data import MetadataCatalog
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY,  Res5ROIHeads

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from .box_predictor import ClipFastRCNNOutputLayersStyle
from detectron2.layers import ShapeSpec
import torch.nn.functional as F 


class ClipRes5ROIHeadsStyle(Res5ROIHeads):   
    def __init__(self, cfg, input_shape) -> None:
        super().__init__(cfg, input_shape)
        clsnames = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes").copy()

        # import pdb;pdb.set_trace()
        ##change the labels to represent the objects correctly
        for name in  cfg.MODEL.RENAME:
            ind = clsnames.index(name[0])
            clsnames[ind] = name[1]
       
        out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * (2 ** 3) ### copied 
        self.box_predictor = ClipFastRCNNOutputLayersStyle(cfg, ShapeSpec(channels=out_channels, height=1, width=1), clsnames)
        # self.clip_im_predictor = self.box_predictor.cls_score # should call it properly
        self.device = cfg.MODEL.DEVICE
    
    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.clip_vb(x)    
        
        
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        style_id= None):
        
        
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets) # 返回512个采样的box
        
        del targets    
        
        proposal_boxes = [x.proposal_boxes for x in proposals]
        num_proposal_boxes = torch.tensor([len(p) for p in proposal_boxes]).to(features['res4'].device)
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes)
        attn_feat = self.attention_global_pool(box_features)    
        
       
        if self.training:
            predictions = self.box_predictor([attn_feat,box_features.mean(dim=(2,3))],num_proposal_boxes,style_id)
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            return losses

        else:
            predictions = self.box_predictor([attn_feat,box_features.mean(dim=(2,3))])
            # inference only return the domian prompt id #########################
            # pred_instances = self.box_predictor.inference(predictions, proposals) 
            scores = F.softmax(predictions,dim=-1)
            scores_split = torch.chunk(scores,19,dim=1)
            scores_all_domian = torch.cat([torch.sum(c, dim=1, keepdim=True) for c in scores_split],dim=1)
            score_mean = torch.mean(scores_all_domian,dim=0)
            domain_prompt_id = torch.argmax(score_mean)
            return domain_prompt_id