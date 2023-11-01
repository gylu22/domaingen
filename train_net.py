from cgi import parse_multipart
import os
import logging
import time
from collections import OrderedDict, Counter
import copy 

import numpy as np

import torch
from torch import autograd
import torch.utils.data as torchdata
from fvcore.nn.precise_bn import get_bn_modules

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.layers.batch_norm import FrozenBatchNorm2d
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_setup
from detectron2.engine import default_argument_parser, hooks, HookBase
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping, build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader, build_detection_test_loader, get_detection_dataset_dicts
from detectron2.data.common import  DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
from detectron2.utils.events import  get_event_storage

from detectron2.utils import comm
from detectron2.evaluation import COCOEvaluator, verify_results, inference_on_dataset, print_csv_format

from detectron2.solver import LRMultiplier
from detectron2.modeling import build_model
from detectron2.structures import  pairwise_iou, Boxes

from fvcore.common.param_scheduler import ParamScheduler
from fvcore.common.checkpoint import Checkpointer

from data.datasets import builtin

from detectron2.evaluation import   COCOEvaluator, inference_on_dataset

from detectron2.data import build_detection_train_loader, MetadataCatalog
import torch.utils.data as data
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as detT

import torchvision.transforms as T
import torchvision.transforms.functional as tF

from modeling import add_stn_config
from modeling import CustomPascalVOCDetectionEvaluator

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logger = logging.getLogger("detectron2")

def setup(args):
    cfg = get_cfg()
    add_stn_config(cfg)
    #hack to add base yaml 
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(model_zoo.get_config_file(cfg.BASE_YAML))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg

class CustomDatasetMapper(DatasetMapper):
    def __init__(self,cfg,is_train) -> None:
        super().__init__(cfg,is_train)
        self.with_crops = cfg.INPUT.CLIP_WITH_IMG
        self.with_random_clip_crops = cfg.INPUT.CLIP_RANDOM_CROPS
        self.with_jitter = cfg.INPUT.IMAGE_JITTER
        self.cropfn = T.RandomCrop#T.RandomCrop([224,224])
        self.aug = T.ColorJitter(brightness=.5, hue=.3)
        self.crop_size = cfg.INPUT.RANDOM_CROP_SIZE

    def __call__(self,dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = detT.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )
       
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        if self.with_jitter:
            dataset_dict["jitter_image"] = self.aug(dataset_dict["image"])
            
        if self.with_crops:
            bbox = dataset_dict['instances'].gt_boxes.tensor
            csx = (bbox[:,0] + bbox[:,2])*0.5
            csy = (bbox[:,1] + bbox[:,3])*0.5
            maxwh = torch.maximum(bbox[:,2]-bbox[:,0],bbox[:,3]-bbox[:,1])
            crops = list()
            gt_boxes = list()
            mean=[0.48145466, 0.4578275, 0.40821073]
            std=[0.26862954, 0.26130258, 0.27577711]    
            for cx,cy,maxdim,label,box in zip(csx,csy,maxwh,dataset_dict['instances'].gt_classes, bbox):

                if int(maxdim) < 10:
                    continue
                x0 = torch.maximum(cx-maxdim*0.5,torch.tensor(0))
                y0 = torch.maximum(cy-maxdim*0.5,torch.tensor(0))
                try:
                    imcrop = T.functional.resized_crop(dataset_dict['image'],top=int(y0),left=int(x0),height=int(maxdim),width=int(maxdim),size=224)
                    imcrop = imcrop.flip(0)/255 # bgr --> rgb for clip
                    imcrop = T.functional.normalize(imcrop,mean,std)
                    # print(x0,y0,x0+maxdim,y0+maxdim,dataset_dict['image'].shape)
                    # print(imcrop.min(),imcrop.max() )
                    gt_boxes.append(box.reshape(1,-1))
                except Exception as e:
                    print(e)
                    print('crops:',x0,y0,maxdim)
                    exit()
                # crops.append((imcrop,label))
                crops.append(imcrop.unsqueeze(0))
            
            if len(crops) == 0:
                dataset_dict['crops'] = []
            else:
                dataset_dict['crops'] = [torch.cat(crops,0),Boxes(torch.cat(gt_boxes,0))]

        if self.with_random_clip_crops:
            crops = []
            rbboxs = []
            
            for i in range(15):
                p = self.cropfn.get_params(dataset_dict['image'],[self.crop_size,self.crop_size])
                c = tF.crop(dataset_dict['image'],*p)
                if self.crop_size != 224:
                    c = tF.resize(img=c,size=224)
                crops.append(c)
                rbboxs.append(p)
            
            crops = torch.stack(crops)
            dataset_dict['randomcrops'] = crops

            #apply same crop bbox to the jittered image
            if self.with_jitter:
                jitter_crops = []
                for p in rbboxs:
                    jc = tF.crop(dataset_dict['jitter_image'],*p) 
                    if self.crop_size != 224:
                        jc = tF.resize(img=jc,size=224)
                    jitter_crops.append(jc)
           
                jcrops = torch.stack(jitter_crops)
                dataset_dict['jitter_randomcrops'] = jcrops

         

        return dataset_dict
    
class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if MetadataCatalog.get(dataset_name).evaluator_type == 'pascal_voc':
            return CustomPascalVOCDetectionEvaluator(dataset_name)
        else:
            return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    
    @classmethod
    def build_train_loader(cls,cfg):
        original  = cfg.DATASETS.TRAIN
        print(original)
        cfg.DATASETS.TRAIN=(original[0],)
        data_loader = build_detection_train_loader(cfg, mapper=CustomDatasetMapper(cfg, True))
        return data_loader
    
    def run_step(self):
        """
        Implement the standard training logic described above.
        """


        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad() 
        losses.backward()   
        self.optimizer.step()
        self._trainer._write_metrics(loss_dict, data_time)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        def do_test_st(flag):
            if flag == 'st':
                model = self.model 
            else:
                print("Error in the flag")

            results = OrderedDict()
            for dataset_name in self.cfg.DATASETS.TEST:
                data_loader = build_detection_test_loader(self.cfg, dataset_name)
                evaluator = CustomPascalVOCDetectionEvaluator(dataset_name)
                results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i
                if comm.is_main_process():
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)
                    storage = get_event_storage()
                    storage.put_scalar(f'{dataset_name}_AP50', results_i['bbox']['AP50'],smoothing_hint=False)
            if len(results) == 1:
                results = list(results.values())[0]
            return results
        
       
        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_SAVE_PERIOD, lambda flag='st': do_test_st(flag)))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
    

class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer=None, scheduler=None):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.
        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
            )
        self._best_param_group_id1 = LRScheduler.get_best_param_group_id(self._optimizer)
    

    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        lr1 = self._optimizer.param_groups[self._best_param_group_id1]["lr"]
        self.trainer.storage.put_scalar("lr", lr1, smoothing_hint=False)

        self.scheduler.step()

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)
      

def custom_build_detection_test_loader(cfg,dataset_name,mapper=None):

    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
   
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    sampler = None
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    collate_fn  = None

    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch

    return torchdata.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def do_test(cfg, model, model_type=''):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = CustomPascalVOCDetectionEvaluator(dataset_name)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        if len(results) == 1:       
            results = list(results.values())[0]
    return results

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg,model)
    trainer = Trainer(cfg)
    ################## temp to freeze the new module ################## 
    # for name, parameter in trainer.model.named_parameters():
    #     if 'cls_score' in name:
    #         parameter.requires_grad = False
    #     elif 'backbone' in name:
    #         parameter.requires_grad = False
    # for name, parameter in trainer.model.named_parameters():
    #     if 'backbone' in name:
    #         parameter.requires_grad = False
    # for name, parameter in trainer.model.named_parameters():
    #     if 'backbone.enc.layer4' in name:  
    #         parameter.requires_grad = True
        
    # with open('grad.txt','w') as f:
    #     for name, parameter in trainer.model.named_parameters():
    #         f.write(f"{name}:{parameter.requires_grad}\n")  
    trainer.resume_or_load(resume=args.resume)
    for dataset_name in cfg.DATASETS.TEST:
        if 'daytime_clear_test' in dataset_name :
            trainer.register_hooks([
                    hooks.BestCheckpointer(cfg.TEST.EVAL_SAVE_PERIOD,trainer.checkpointer,f'{dataset_name}_AP50',file_prefix='model_best'),
                    ])
    autograd.set_detect_anomaly(True)
    trainer.train()

    
# def main(args):
#     cfg = setup(args)
      
#     if args.eval_only:
        
#         model = Trainer.build_model(cfg)
#         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#             cfg.MODEL.WEIGHTS, resume=args.resume)
#         model.training = False
#         model.style_head.training = False
#         # poolar = ROIPooler()
#         for dataset_name in cfg.DATASETS.TEST:
#             data_loader = build_detection_test_loader(cfg, dataset_name)
#             data = iter(data_loader)
#             feature_maps = []
#             feature_style = []
#             # stds = []
#             for i in range(20):
#                 batch_data = next(data)
#                 # if i < 20:
#                 #     continue
#                 feature_map,feature_style = model(batch_data) 
#                 # feature_map = feature_map['res4']
#                 # std = torch.std(feature_map, dim=(-2, -1))
#                 feature_map = torch.mean(feature_map, dim=(-2, -1))
#                 feature_style = torch.mean(feature_style, dim=(-2, -1))
#                 # feature_map = poolar(feature_map)
#                 feature_maps.append(feature_map)
#                 feature_style.append(feature_style)
#                 # stds.append(std)
#             feature_maps = torch.cat(feature_maps,dim=0)
#             feature_style = torch.cat(feature_style,dim=0)
#             # stds = torch.cat(stds,dim=0)        
#             # torch.save(feature_maps,f"inter_features/mean/{dataset_name}_20-40.pkl")
#             torch.save(feature_maps,f"inter_features/norm/{dataset_name}.pkl")
#             torch.save(feature_style,f"inter_features/norm/{dataset_name}_style.pkl")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    print("Command Line Args:", args)

    main(args)
