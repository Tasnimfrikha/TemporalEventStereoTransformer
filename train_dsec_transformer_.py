from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.ours_large_transformer import ours_large
from utils import *
from dataloader.dsceDataset import get_datasets
from dataloader.dsecProvider_train import DatasetProvider
from dataloader.contrast.raw_event_utils import custom_collate

from utils.evaluation import do_evaluation
from utils.flow import EventWarping

import argparse
import os
import numpy as np
import time
from tqdm.auto import tqdm
from shutil import copyfile
import yaml
import cv2
import matplotlib.pyplot as plt
import warnings
import pdb
import threading
import flow_vis
import pathlib
from pathlib import Path
from collections import OrderedDict

def custom_collate(batch):
    n_frames = len(batch[0])  # i.e. len(frame_idxs) = 4
    output = []
    for t in range(n_frames):
        left_voxels = torch.stack([item[t][0] for item in batch])
        right_voxels = torch.stack([item[t][1] for item in batch])
        disparities = torch.stack([item[t][2] for item in batch])
        pads = [item[t][3] for item in batch]
        debug_infos = [item[t][4] for item in batch]
        output.append((left_voxels, right_voxels, disparities, pads, debug_infos))
    return output


class EventStereo():
    def __init__(self, config, args):
        self.config = config
        self.args = args
        
        # Enforce GPU-only execution
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA-enabled GPU detected. This script requires a GPU.")
        self.device = torch.device("cuda")
        
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        
        self.epoch = int(config['model']['epoch'])
        self.frame_idxs = config['model']['frame_idxs']
        self.use_prev_gradient = config['model']['use_prev_gradient']
        
        if isinstance(self.frame_idxs, list):
            self.frame_idxs.sort()
        elif isinstance(self.frame_idxs, str):
            self.frame_idxs = eval(self.frame_idxs)
            assert len(self.frame_idxs) > 0
            for i in self.frame_idxs:
                assert isinstance(i, int)
            
        # Make dataloaders
        data_root = config['model']['dataset_root_path']
        height = config['model']['height']
        width = config['model']['width']
        in_ch = config['model']['in_ch']
        
        self.orig_height = config['model']['orig_height']
        self.orig_width = config['model']['orig_width']
        maxdisp = config['model']['maxdisp']
        self.eval_maxdisp = config['model']['eval_maxdisp']
        
        self.use_pseudo_gt = config['model'].get('use_pseudo_gt', False)
        self.use_disp_gt_mask = config['model'].get('use_disp_gt_mask', False)
        self.use_mini_data = config['model'].get('use_mini_data', False)
        self.use_super_mini_data = config['model'].get('use_super_mini_data', False)
        
        self.val_of_viz = config['model'].get('val_of_viz', False)
        
        pseudo_root_path = config['model']['pseudo_root_path']
        self.use_raw_provider = config['model'].get('use_raw_provider', False)
        
        data_root = config['model']['dataset_root_path']
        raw_data_root = config['model']['dataset_raw_root_path']
        
        self.use_disp_flow_warp_mask = config['model'].get('use_disp_flow_warp_mask', False)
        self.use_featuremetric_loss = config['model'].get('use_featuremetric_loss', False)
        self.use_disp_loss = config['model'].get('use_disp_loss', True)
        self.use_contrast_loss = config['model'].get('use_contrast_loss', False)
        self.use_stereo_loss = config['model'].get('use_stereo_loss', True)
        
        dataset_provider = DatasetProvider(
            Path(data_root), Path(raw_data_root), frame_idxs=self.frame_idxs, 
            eval_maxdisp=self.eval_maxdisp, num_bins=in_ch, pad_width=width, pad_height=height,
            pseudo_path=pseudo_root_path, use_mini=self.use_mini_data, use_super_mini=self.use_super_mini_data
            valid_sequence_file="/home/cerbere-25/TemporalEventStereo_Official/valid_sequences.txt"

        )
        self.train_dataset = dataset_provider.get_train_dataset()
        self.validation_dataset = dataset_provider.get_val_dataset()
        
        self.train_dataset[0]
        self.validation_dataset[0]
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=config["train"]["shuffle"],
            num_workers=config["train"]["num_worker"],
            drop_last=False,
            pin_memory=True,
            collate_fn=custom_collate  # Add custom collate function
        )
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=config["validation"]["batch_size"],
            shuffle=config["validation"]["shuffle"],
            num_workers=config["validation"]["num_worker"],
            drop_last=False,
            pin_memory=False,
            collate_fn=custom_collate  # Add custom collate function
        )
        
        # Define model
        model_name = config['model']['type']
        self.model_name = model_name
        
        if model_name == 'ours_large_transformer':
            self.model = ours_large(maxdisp, self.orig_height, self.orig_width, in_ch=in_ch)
        else:
            raise Exception('Wrong model type')
        
        self.L1_lossFn = nn.L1Loss().to(self.device)
        self.contrast_lossFn = EventWarping(
            (self.orig_height, self.orig_width), 
            {'flow_regul_weight': config['model']['flow_smooth_weight']}
        ).to(self.device)
        self.flow_scale = config['model']['flow_scale']
        
        # Move model to GPU
        self.model = nn.DataParallel(self.model).to(self.device)

        # Add logger
        if 'debug' in args.savemodel:
            os.makedirs(args.savemodel, exist_ok=True)
        else:
            os.makedirs(args.savemodel, exist_ok=True)
        os.makedirs(os.path.join(args.savemodel, 'prediction'), exist_ok=True)
        os.makedirs(os.path.join(args.savemodel, 'warped'), exist_ok=True)
        os.makedirs(os.path.join(args.savemodel, 'flow'), exist_ok=True)
        copyfile(args.config, os.path.join(args.savemodel, 'config.yaml'))
        model_file_path = os.path.join('models', config['model']['type']+'.py')
        if os.path.isfile(model_file_path):
            copyfile(model_file_path, os.path.join(args.savemodel, config['model']['type']+'.py'))
        self.logger = Logger(args.savemodel)
        self.logger.log_and_print("Model name: " + model_name)
        self.logger.log_and_print("Log path: " + args.savemodel + "\n")
        tensorboard_path = os.path.join(args.savemodel, 'runs')
        self.writer = SummaryWriter(tensorboard_path)
        self.log_train_every_n_batch = int(config['log']['log_train_every_n_batch'])
        self.save_test_every_n_batch = int(config['log']['save_test_every_n_batch'])
        lr = self.config['lr']
        
        do_not_load_layer = config['model'].get('do_not_load_layer', [])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler_epoch = self.epoch if self.epoch != 0 else 1

        loadmodel = config['model']['load_model']
        load_strict = config['model']['load_strict']
        load_optim = config['model']['load_optim']
        remove_feat_first_weight = config['model']['remove_feat_first_weight']
        
        if loadmodel != '':
            if os.path.isfile(loadmodel):
                self.logger.log_and_print('Load pretrained model from '+ loadmodel)
                pretrain_dict = torch.load(loadmodel, map_location=self.device)
                if remove_feat_first_weight:
                    del(pretrain_dict['state_dict']['module.feature_extraction.firstconv.0.0.weight'])
                
                new_state_dict = OrderedDict()
                for k, v in pretrain_dict['state_dict'].items():
                    name = k.split('.')[1]
                    if name not in do_not_load_layer:
                        new_name = k
                        new_state_dict[new_name] = v
                    else:
                        print(k)
                self.model.load_state_dict(new_state_dict, strict=load_strict)
                
                if ('optimizer' in pretrain_dict) & load_optim:
                    self.optimizer.load_state_dict(pretrain_dict['optimizer'])
                else:
                    warnings.warn("Warning: Load state dict does not contain optimizer state, might cause different result!!!")
            else:
                raise Exception('Cannot find pretrained file')
            
        pretrain_freeze = config['model']['pretrain_freeze']
        model_layer_name = set()
        if pretrain_freeze:
            for name, param in self.model.named_parameters():
                layer_name = name.split('.')[1]
                if layer_name not in ['dres4', 'classif3', 'of_block', 'fusion']:
                    param.requires_grad = False
                else:
                    model_layer_name.add(layer_name)
            print("trained parameter: ", model_layer_name)
        
        self.logger.log_and_print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))
        self.logger.log_and_print(f'Running on device: {self.device}')
        self.logger.log_and_print('Done initialization\n')
        
    def remove_padding(self, image, pad):
        image_is_tensor = isinstance(image, torch.Tensor)
        if image_is_tensor:
            image = [image]

        # Gestion du format [ [(left, right, top, bottom)] ]
        if isinstance(pad, (list, tuple)) and len(pad) == 1 and isinstance(pad[0], (list, tuple)):
            pad = pad[0]

        if not isinstance(pad, (list, tuple)) or len(pad) != 4:
            raise ValueError(f"Expected pad to be a 4-element tuple or list, got {pad}")

        # Extraire les valeurs (gérer les cas où c’est par exemple [[0], [0], [0], [0]])
        pad_left = pad[0][0] if isinstance(pad[0], (list, tuple)) else pad[0]
        pad_right = pad[1][0] if isinstance(pad[1], (list, tuple)) else pad[1]
        pad_top = pad[2][0] if isinstance(pad[2], (list, tuple)) else pad[2]
        pad_bottom = pad[3][0] if isinstance(pad[3], (list, tuple)) else pad[3]

        full_h, full_w = image[0].shape[-2:]
        h = full_h - pad_top - pad_bottom
        w = full_w - pad_left - pad_right

        disp_no_pad = []
        for disp in image:
            assert disp.shape[-2] == full_h and disp.shape[-1] == full_w
            disp_no_pad.append(disp[..., pad_top:full_h - pad_bottom, pad_left:full_w - pad_right])

        return disp_no_pad[0] if image_is_tensor else disp_no_pad

        
    def calculate_contrast_loss(self, flow, e_list, pol_mask):
        loss = 0.0
        device = flow.device
        for b in range(flow.shape[0]):
            batch_flow = [flow[b].unsqueeze(0)]
            loss += self.contrast_lossFn(
                batch_flow, 
                e_list[b].unsqueeze(0).to(device), 
                pol_mask[b].unsqueeze(0).to(device)
            )
        return loss
    
    def train(self, batch):
        self.model.train()
        
        total_disp_loss = torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device) 
        
        for i, (imgL, imgR, dispL, pad, batch_debug) in enumerate(batch):
            frame_idx = self.frame_idxs[i]
            if imgL.ndim == dispL.ndim:
                dispL = dispL.squeeze(axis=1)
            if self.use_pseudo_gt:
                pseudo_gt = batch_debug[-1].get('pseudo_disp', None) 
                pseudo_gt = pseudo_gt.to(self.device)
            
            imgL, imgR, disp_true = imgL.to(self.device), imgR.to(self.device), dispL.to(self.device)
            disp_true = disp_true.type(torch.float32)
            mask = disp_true <= self.eval_maxdisp
            mask.detach_()
            self.optimizer.zero_grad()
            
            if self.model_name in ['stackhourglass', 'stackhourglass_small', 'stackhourglass_v2']:
                output1, output2, output3 = self.model(imgL, imgR)
                output1 = torch.squeeze(output1, 1)
                output2 = torch.squeeze(output2, 1)
                output3 = torch.squeeze(output3, 1)
                output1, output2, output3, disp_true, mask = self.remove_padding([output1, output2, output3, disp_true, mask], pad)
                total_loss = (
                    0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], )
                    + 0.7 * F.smooth_l1_loss(output2[mask], disp_true[mask], )
                    + F.smooth_l1_loss(output3[mask], disp_true[mask], )
                )
            elif self.model_name in ['ours_large_transformer']:
                if self.use_pseudo_gt:
                    gt_nan_mask = torch.isnan(disp_true)
                    pseudo_gt = pseudo_gt.to(self.device)

                    # Fix si pseudo_gt est 2D
                    if pseudo_gt.ndim == 2:
                        pseudo_gt = pseudo_gt.unsqueeze(0).repeat(disp_true.size(0), 1, 1)  # B = disp_true.shape[0]
                    elif pseudo_gt.ndim == 4 and pseudo_gt.shape[1] == 1:
                        pseudo_gt = pseudo_gt.squeeze(1)  # [B, 1, H, W] → [B, H, W]

                    assert pseudo_gt.shape == disp_true.shape, f"Shape mismatch: pseudo_gt {pseudo_gt.shape}, disp_true {disp_true.shape}"

                    disp_true[gt_nan_mask] = pseudo_gt[gt_nan_mask]

                small_disp_true = F.interpolate(disp_true.unsqueeze(1), scale_factor=0.25, mode="bilinear", align_corners=True) / 4.0
                small_disp_true = small_disp_true.squeeze(1)
                if i == 0:
                    prev_feat = None
                    prev_disp = None
                    prev_pred = None
                    prev_cost = None
                
                if frame_idx < 0 and (not self.use_prev_gradient or i == 0):
                    self.model.eval()
                    output, prev_feat, debug, cost_volume = self.model(imgL, imgR, prev_feat, prev_disp, prev_cost, prev_pred)
                    prev_disp = small_disp_true
                    prev_cost = cost_volume
                    prev_pred = output.detach()
                    self.model.train()

                else:
                    output1, output2, output3, prev_feat, debug, temp_warp_disp, cost_volume = self.model(
                        imgL, imgR, prev_feat, prev_disp, prev_cost, prev_pred
                    )
                    temp_warp_disp = torch.squeeze(temp_warp_disp, 1)
                    prev_disp = small_disp_true
                    prev_cost = cost_volume
                    prev_pred = output3.detach()
                    output1 = torch.squeeze(output1, 1)
                    output2 = torch.squeeze(output2, 1)
                    output3 = torch.squeeze(output3, 1)
                    
                    output1, output2, output3, disp_true, mask = self.remove_padding([output1, output2, output3, disp_true, mask], pad)
                    
                    loss = (
                        0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask])
                        + 0.7 * F.smooth_l1_loss(output2[mask], disp_true[mask])
                        + 1.0 * F.smooth_l1_loss(output3[mask], disp_true[mask])
                    )

                    

                    
                    small_mask = (small_disp_true <= (self.eval_maxdisp / 4.0)) & (small_disp_true > 0)
                    warp_mask = (temp_warp_disp <= (self.eval_maxdisp / 4.0)) & (temp_warp_disp > 0)
                    warp_mask = warp_mask & small_mask
                    disp_loss = 0.1 * F.smooth_l1_loss(temp_warp_disp[warp_mask], small_disp_true[warp_mask])
                    loss = loss + disp_loss
            else:
                raise NotImplementedError
            if torch.isnan(loss):
                print("⚠️ La loss est NaN, on saute ce batch.")
                continue

        loss.backward()
        self.optimizer.step()

        return loss.data.cpu(), total_disp_loss.data.cpu()
    
    def test(self, batch):
        self.model.eval()
        debug_batch = []
        img_batch = []
        disp_true_batch = []
        debug = {}
        total_batch_time = 0.0
        
        for i, (imgL, imgR, dispL, pad, data_debug) in enumerate(batch):
            frame_idx = self.frame_idxs[i]
            if imgL.ndim == dispL.ndim:
                dispL = dispL.squeeze(axis=1)
            
            imgL, imgR, disp_true = imgL.to(self.device), imgR.to(self.device), dispL.to(self.device)
            disp_true = disp_true.type(torch.float32)
            
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            with torch.no_grad():
                if self.model_name in ['ours_large']:
                    small_disp_true = F.interpolate(disp_true.unsqueeze(1), scale_factor=0.25, mode="bilinear", align_corners=True) / 4.0
                    small_disp_true = small_disp_true.squeeze(1)
                    if i == 0:
                        prev_feat = None
                        prev_pred = None
                        prev_cost = None
                        prev_disp = None
                    output3, prev_feat, debug, cost_volume = self.model(imgL, imgR, prev_feat, prev_disp, prev_cost, prev_pred)
                    prev_pred = output3
                    prev_cost = cost_volume
                    debug['prev_gt'] = prev_disp
                    prev_disp = small_disp_true
                else:
                    output3 = self.model(imgL, imgR)
                output3 = torch.squeeze(output3[0], dim=1)
            ender.record()
            torch.cuda.synchronize()
            total_batch_time += starter.elapsed_time(ender)

            img, disp_true = self.remove_padding([output3, disp_true], pad[0])

            mask = disp_true <= self.eval_maxdisp
            if len(disp_true[mask]) == 0 or frame_idx < 0:
                loss = 0
            else:
                loss = F.l1_loss(img[mask], disp_true[mask])
                
            for key in debug:
                if isinstance(debug[key], torch.Tensor):
                    if self.orig_width != debug[key].shape[-1]:
                        debug[key] = self.remove_padding(debug[key], pad)
            data_debug[-1].update(debug)
            debug_batch.append(data_debug)
            
            img_batch.append(img)
            disp_true_batch.append(disp_true)

        return (loss.data.cpu(), img_batch, disp_true_batch, total_batch_time / len(batch), debug_batch)

    def run(self):
        start_full_time = time.time()
        validation_result = []
        for epoch in range(0, self.epoch):
            self.logger.log_and_print('This is %d-th epoch' % (epoch))
            start = time.perf_counter()
            total_train_loss = 0
            total_disp_loss = 0


            ## Training ##
            for batch_idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                if batch_idx > 3:
                    break
                print("DEBUG frame_idxs:", self.frame_idxs)
                print("DEBUG batch len:", len(batch))
                assert len(batch) == len(self.frame_idxs)
                start_time = time.time()

                loss, disp_loss = self.train(batch)
                if batch_idx % 400 == 0:
                    self.logger.log_and_print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
                if batch_idx % self.log_train_every_n_batch == 0:
                    self.writer.add_scalar('Loss/train', loss, epoch * len(self.train_loader) + batch_idx)
                total_train_loss += loss
                total_disp_loss += disp_loss
            self.logger.log_and_print('epoch %d total training loss = %.3f, disp loss = %.3f' % (
                epoch, total_train_loss / len(self.train_loader), total_disp_loss / len(self.train_loader)))
            finish = time.perf_counter()
            self.logger.log_and_print(f'Train finished in {round(finish - start, 4)} second(s)')
            
            # Save checkpoint
            savefilename = self.args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'train_loss': total_train_loss / len(self.train_loader),
                'optimizer': self.optimizer.state_dict(),
            }, savefilename)
            
            # Validation #
            total_val_loss = 0
            disp_pred = []
            disp_gt = []
            total_batch_time = 0
            total_idx = 0
            total_photo_loss = 0.0
            for batch_idx, batch in tqdm(enumerate(self.validation_loader), total=len(self.validation_loader)):
                assert len(batch) == len(self.frame_idxs)
                val_loss, disp_pred_batch, disp_gt_batch, batch_time, debug = self.test(batch)
                disp_pred.append(disp_pred_batch[-1].cpu())
                disp_gt.append(disp_gt_batch[-1].cpu())
                total_val_loss += val_loss
                total_batch_time += batch_time
                total_idx += 1
                if isinstance(debug, list) and len(debug) > 0 and isinstance(debug[-1], dict):
                    total_photo_loss += debug[-1].get('photo_loss', 0.0) if isinstance(debug, list) and len(debug) > 0 and isinstance(debug[-1], dict) else 0.0
                else:
                    total_photo_loss += 0.0

                
                if self.val_of_viz:
                    if batch_idx % self.save_test_every_n_batch == 0:
                        sequence_name = debug[-1]['left_event_path'][0].split('/')[-4]
                        sequence_number = debug[-1]['left_event_path'][0][-10:-4]
                        up_left_of = debug[-1].get('left_of', None)[0].detach().cpu().permute(1, 2, 0).numpy()
                        flow_color = flow_to_color(up_left_of, clip_flow=10.0, convert_to_bgr=True)
                        cv2.imwrite(os.path.join(self.args.savemodel, 'flow', sequence_name + '_' + sequence_number + '.png'), flow_color)
            
            finish = time.perf_counter()
            self.logger.log_and_print(f'Validation finished in {round(finish - start, 4)} second(s)')
            self.logger.log_and_print('Average validation time per batch = {:.3f}'.format(total_batch_time / total_idx))
            error_dict = do_evaluation(torch.cat(disp_pred, dim=0), torch.cat(disp_gt, dim=0), 0.0, self.eval_maxdisp)
            self.logger.log_and_print('total validation loss = %.3f' % (total_val_loss / len(self.validation_loader)))
            self.logger.log_and_print('total RGB photo loss = %.5f' % (total_photo_loss / len(self.validation_loader)))
            self.logger.log_and_print(error_dict)
            validation_result.append(error_dict['1px'].item())

        self.logger.log_and_print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
        self.logger.log_and_print('############ TRAINING END ############')
        validation_result.append(100.0)
        best_val_epoch = np.argmin(validation_result)
        self.logger.log_and_print('Best validation result is {0:0.4f} at epoch {1:d}'.format(validation_result[best_val_epoch], best_val_epoch))
        
        # Testing phase
        total_test_loss = 0
        disp_pred = []
        disp_gt = []
        total_batch_time = 0
        total_idx = 0
        total_photo_loss = 0.0
        for batch_idx, batch in tqdm(enumerate(self.validation_loader), total=len(self.validation_loader), 
                                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', dynamic_ncols=True):
            assert len(batch) == len(self.frame_idxs)
            test_loss, disp_pred_batch, disp_gt_batch, batch_time, debug = self.test(batch)
            total_test_loss += test_loss
            total_batch_time += batch_time
            total_idx += 1

            if batch_idx % self.save_test_every_n_batch == 0:
                sequence_name = debug[-1]['left_event_path'][0].split('/')[-4]
                sequence_number = debug[-1]['left_event_path'][0][-10:-4]
                save_matrix(
                    os.path.join(self.args.savemodel, 'prediction', sequence_name + '_' + sequence_number + '.png'),
                    disp_pred_batch[-1][0].detach().cpu() * 7.0,
                    0.0, 255.0, 'gray', False
                )
            torch.cuda.synchronize()
        
        print(self.args.savemodel)
    
def main():
    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('--config', default='/home/cerbere-25/TemporalEventStereo_Official/config/dsec_train.yaml',
                        help='config file path')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--savemodel', default='./',
                        help='save model')
    args = parser.parse_args()
    assert os.path.isfile(args.config), f"Config file not found: {args.config}"
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    eventstereo = EventStereo(config=config, args=args)
    eventstereo.run()

if __name__ == '__main__':
    main()