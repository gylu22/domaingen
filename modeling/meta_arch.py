from ast import mod
import math
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from typing import Dict,List,Optional

from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.layers import batched_nms
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer
from .style_head import ClipRes5ROIHeadsStyle

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


@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackbone(GeneralizedRCNN):

    def __init__(self,cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.colors = self.generate_colors(7)
        self.backbone.set_backbone_model(self.roi_heads.box_predictor.cls_score.visual_enc)
        print('done!')
    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        clip_images = [x["image"].to(self.pixel_mean.device) for x in batched_inputs]
        mean=[0.48145466, 0.4578275, 0.40821073]
        std=[0.26862954, 0.26130258, 0.27577711] 
  

        clip_images = [ T.functional.normalize(ci.flip(0)/255, mean,std) for ci in clip_images]
        clip_images = ImageList.from_tensors(
            [i  for i in clip_images])
        return clip_images


    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0] # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)
        
        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)                
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def generate_colors(self,N):
        import colorsys
        '''
            Generate random colors.
            To get visually distinct colors, generate them in HSV space then
            convert to RGB.
        '''
        brightness = 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
        perm = np.arange(7)
        colors = [colors[idx] for idx in perm]
        return colors

            
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        if detected_instances is None:
            if self.proposal_generator is not None:
                logits,proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
            # boxes = batched_inputs[0]['instances'].gt_boxes.to(images.tensor.device)
            # logits = 10*torch.ones(len(boxes)).to(images.tensor.device)
            # dictp = {'proposal_boxes':boxes,'objectness_logits':logits}
            # new_p = Instances(batched_inputs[0]['instances'].image_size,**dictp)    
            # proposals = [new_p]
             
            try:
                results, _ = self.roi_heads(images, features, proposals, None, None, self.backbone)
            except:
                results, _ = self.roi_heads(images, features, proposals, None, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."

            allresults = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)


            return allresults
        else:
            return results


        
@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackboneGenTrainable(ClipRCNNWithClipBackbone):

    def __init__(self,cfg) -> None:
        super().__init__(cfg)
        domain_text = {}
        with open('prunedprompts2.txt','r') as f:
            for ind,l in enumerate(f):
                domain_text.update({str(ind):l.strip()})
        import clip
        domain_tk = dict([(k,clip.tokenize(t)) for k,t in domain_text.items()])
        self.domain_tk = torch.cat([d for _,d in domain_tk.items()],dim=0).to('cuda')
        # get the domain text embedding only in training
        if self.training:
            clip_model = self.roi_heads.box_predictor.cls_score.model
            with torch.no_grad():
                self.domain_text_features = F.normalize(clip_model.encode_text(self.domain_tk),dim=1)
                
        # define the domain text embedding projection module 
        self.apply_aug = cfg.AUG_PROB
        # PMS: prompt based mean and std generation module 
        # mean linear module 
        self.gen_style_mean = nn.Sequential(nn.Linear(512,1024),
                                      nn.Linear(1024,256),
                                      nn.Sigmoid())
        # std linear module 
        #####
        self.gen_style_std = nn.Sequential(nn.Linear(512,1024),
                                     nn.Linear(1024,256),
                                     nn.ReLU())
        self._init_weights(self.gen_style_mean)
        self._init_weights(self.gen_style_std)
        # style head 
        # self.style_head = ClipRes5ROIHeadsStyle(cfg, self.backbone.output_shape())
        # add CLIP Vb for the head
        # self.style_head.clip_vb = copy.deepcopy(self.backbone.enc.layer4)        
        # add CLIP attention_global_pool module 
        # self.style_head.attention_global_pool = copy.deepcopy(self.backbone.attention_global_pool)
            
             
             
    def _init_weights(self,module):
        for m in module:
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1)
                nn.init.zeros_(m.bias)
         
        
    def forward(self,batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
   
        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0] # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        # get backbone layer2 output tensor 
        style_id  = torch.randint(low=0, high=19, size=(b,)).to(images.device)
        # self.domian_id = style_id
        # generate the mean and std of the transform 
        style_domain = self.domain_text_features[style_id]
        beta = self.gen_style_mean(style_domain)
        gamma = self.gen_style_mean(style_domain)

        features = self.backbone(images.tensor,beta,gamma)
        # apply the instance normlization 
        # random select bs style 
        ## style_id  = torch.randint(low=0, high=19, size=(b,)).to(images.device)
        # self.domian_id = style_id
        # generate the mean and std of the transform 
        ## style_domain = self.domain_text_features[style_id]
        ## beta = self.gen_style_mean(style_domain)
        ## gamma = self.gen_style_mean(style_domain)
        ## gamma = nn.ReLU(gamma)
        
        
        # feature style transform ####################
        # features_ = {}
        # new_features = self.instance_norm(features['res4'],beta,gamma)
        # features_['res4'] = new_features
        
        # features after instance normlization 
        if self.proposal_generator is not None:
            if self.training:
                logits,proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)                
            else:
                logits,proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        # compute the style score and bbox loss 
        # style_loss = self.style_head(images, features_, proposals, gt_instances,style_id)
        # style_loss_ = {}
        # for key,val in style_loss.items():
            # style_loss_[f'style_{key}'] = val
            
        # compute the original cls and bbox loss 
        # try:
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, self.backbone)
        # except Exception as e:
        #     print(e)
        #     _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)
        
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        # losses.update(style_loss_)
        return losses
        
###################### test time feature transform     ############################   
    # def inference(
    #     self,
    #     batched_inputs: List[Dict[str, torch.Tensor]],
    #     detected_instances: Optional[List[Instances]] = None,
    #     do_postprocess: bool = True,
    # ):
        
    #     """
    #     Run inference on the given inputs.
    #     Args:
    #         batched_inputs (list[dict]): same as in :meth:`forward`
    #         detected_instances (None or list[Instances]): if not None, it
    #             contains an `Instances` object per image. The `Instances`
    #             object contains "pred_boxes" and "pred_classes" which are
    #             known boxes in the image.
    #             The inference will then skip the detection of bounding boxes,
    #             and only predict other per-ROI outputs.
    #         do_postprocess (bool): whether to apply post-processing on the outputs.
    #     Returns:
    #         When do_postprocess=True, same as in :meth:`forward`.
    #         Otherwise, a list[Instances] containing raw network outputs.
    #     """
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)
    #     _,proposals_, _ = self.proposal_generator(images, features)
        
    #     style_id = self.style_head(images,features,proposals_)
        
    #     style_feature = self.domain_text_features[style_id].unsqueeze(0)
    #     mean = self.gen_style_mean(style_feature)
    #     std = self.gen_style_std(style_feature)
    #     features['res4'] = self.inverse_instance_norm(features['res4'],mean,std)
       
    #     if detected_instances is None:
    #         if self.proposal_generator is not None:
    #             _,proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]
    #         try:
    #             backbone = self.backbone
    #             results, _ = self.roi_heads(images, features, proposals, None, backbone)
    #         except AttributeError :
    #             _,results, _ = self.roi_heads(images, features, proposals, None, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        

    #     if do_postprocess:
    #         assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."

    #         allresults = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)


    #         return allresults
    #     else:
    #         return results
    
    
    
    # def instance_norm(self,
    #                     x:torch.Tensor,
    #                     beta:torch.Tensor, 
    #                     gamma:torch.Tensor,
    #                     p = 0.5) -> torch.Tensor:
        
    #     if random.random() > p:
    #         return x
    #     B = x.size(0)
    #     eps = 1e-6
    #     mu = x.mean(dim=[2, 3], keepdim=True)
    #     var = x.var(dim=[2, 3], keepdim=True)
    #     sig = (var + eps).sqrt()
    #     mu, sig = mu.detach(), sig.detach()
    #     x_normed = (x-mu) / sig
    #     return x_normed * gamma.view(gamma.shape[0],gamma.shape[1],1,1) + beta.view(beta.shape[0],beta.shape[1],1,1)
    
 
    
    
    # def inverse_instance_norm(self,
    #                             features:torch.Tensor,
    #                             beta:torch.Tensor, 
    #                             gamma:torch.Tensor) -> torch.Tensor:
    #     # assert (features.shape[1] == beta.shape[1])  & (features.shape[0] == gamma.shape[0]),'feature chanel must be the same as mean and std dim!'
        
    #     mean = torch.mean(features, dim=(2, 3), keepdim=True)
    #     std = torch.std(features, dim=(2, 3), keepdim=True)
        
    #     std[std == 0] = 1e-5
    #     gamma[gamma==0] = 1e-5
    #     normalized_features =(features - mean) / std
    #     inverse_features = (normalized_features-beta.view(beta.shape[0],beta.shape[1],1,1)) / gamma.view(gamma.shape[0],gamma.shape[1],1,1)
    #     return inverse_features
    
    
    
    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
        ):
            """
            Run inference on the given inputs.
            Args:
                batched_inputs (list[dict]): same as in :meth:`forward`
                detected_instances (None or list[Instances]): if not None, it
                    contains an `Instances` object per image. The `Instances`
                    object contains "pred_boxes" and "pred_classes" which are
                    known boxes in the image.
                    The inference will then skip the detection of bounding boxes,
                    and only predict other per-ROI outputs.
                do_postprocess (bool): whether to apply post-processing on the outputs.
            Returns:
                When do_postprocess=True, same as in :meth:`forward`.
                Otherwise, a list[Instances] containing raw network outputs.
            """
            assert not self.training

            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            # features = self.backbone.forward_res4(features['res3'])
            
            if detected_instances is None:
                if self.proposal_generator is not None:
                    logits,proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                
                # boxes = batched_inputs[0]['instances'].gt_boxes.to(images.tensor.device)
                # logits = 10*torch.ones(len(boxes)).to(images.tensor.device)
                # dictp = {'proposal_boxes':boxes,'objectness_logits':logits}
                # new_p = Instances(batched_inputs[0]['instances'].image_size,**dictp)    
                # proposals = [new_p]
                
                try:
                    results, _ = self.roi_heads(images, features, proposals, None, self.backbone)
                except:
                    results, _ = self.roi_heads(images, features, proposals, None, None)
            else:
                detected_instances = [x.to(self.device) for x in detected_instances]
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            

            if do_postprocess:
                assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."

                allresults = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)


                return allresults
            else:
                return results