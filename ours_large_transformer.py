from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
from utils import *
import timm


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )
    
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

#class TransformerFlow(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Swin Transformer Tiny backbone
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, features_only=True)

        # Adapte le nombre de canaux en entrée à ce que Swin attend (3)
        self.input_proj = nn.Conv2d(128, 3, kernel_size=1)

        # Output projection pour obtenir les 4 flux
        self.output_proj = nn.Conv2d(self.swin.feature_info[-1]['num_chs'], 4, kernel_size=1)

    def forward(self, x):  # x: (B, 256, H, W)
        B, C, H, W = x.shape

        # Réduction de l’entrée à 3 canaux pour compatibilité avec Swin
        x = self.input_proj(x)  # (B, 3, H, W)

        # Passage dans Swin Transformer
        feats = self.swin(x)  # List of 4 feature maps (multi-scale)
        x = feats[-1]         # On prend la dernière couche (plus profonde)

        # Upsample à la taille originale
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        # Projection finale pour prédire les 4 canaux (flow)
        return self.output_proj(x)  # (B, 4, H, W)
    
class TransformerFlow(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Conv2d(128, embed_dim, 1)
        self.output_proj = nn.Conv2d(embed_dim, 4, 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):  # x: (B, 256, H, W)
        B, C, H, W = x.shape
        x = self.input_proj(x)                    # (B, 64, H, W)
        x_flat = x.flatten(2).transpose(1, 2)     # (B, N, 64)

        # Encodage temporel binaire : 0 pour t, 1 pour t-1
        t_encoding = torch.zeros_like(x_flat)
        t_encoding[:, x_flat.size(1)//2:, :] = 1.0
        x_encoded = x_flat + t_encoding

        x_encoded = self.encoder(x_encoded)       # (B, N, 64)
        x_out = x_encoded.transpose(1, 2).reshape(B, 64, H, W)
        return self.output_proj(x_out)            # (B, 4, H, W)


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)
        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes))

    def forward(self, x ,presqu, postsqu):
        out  = self.conv1(x)
        pre  = self.conv2(out)
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)
        out  = self.conv3(pre)
        out  = self.conv4(out)
        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True)
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True)
        out  = self.conv6(post)
        return out, pre, post

class TESNet(nn.Module):
    def __init__(self, maxdisp, height, width, in_ch = 5):
        super(TESNet, self).__init__()
        self.maxdisp = maxdisp
        self.img_size = [height, width]
        self.feature_extraction = feature_extraction(in_ch)
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.dres4 = hourglass(32)
        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))
        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))
        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))
        self.of_block = TransformerFlow()
        self.fusion = nn.Sequential(convbn(64, 32, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.cost_squeeze_block = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right, prev_feat=None, prev_gt_disp=None, prev_cost = None, prev_pred = None, curr_gt_disp=None):
        B, C, H, W = left.size()
        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        # ASSIGN BEFORE USING
        if prev_feat is None:
            prev_imgL = torch.zeros_like(refimg_fea)
            prev_imgR = torch.zeros_like(targetimg_fea)
        else:
            prev_imgL, prev_imgR = prev_feat

        four_of = self.of_block(torch.cat([refimg_fea, targetimg_fea, prev_imgL, prev_imgR], dim=1))

        up_flow = F.interpolate(four_of, scale_factor = 4, mode="bilinear", align_corners=False)
        up_flow = up_flow * 4
        left_of = torch.stack([four_of[:, 0], four_of[:,2]], dim=1)
        right_of = torch.stack([four_of[:, 1], four_of[:,3]], dim=1)
        up_left_of = torch.stack([up_flow[:, 0], up_flow[:,2]], dim=1)
        up_right_of = torch.stack([up_flow[:, 1], up_flow[:,3]], dim=1)
        prev_imgL = flow_warp(prev_imgL, left_of)
        prev_imgR = flow_warp(prev_imgR, right_of)
        refimg_fea = self.fusion(torch.cat([refimg_fea, prev_imgL], dim=1))
        targetimg_fea = self.fusion(torch.cat([targetimg_fea, prev_imgR], dim=1))
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        for i in range(self.maxdisp//4):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1+cost0
        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost1 = torch.squeeze(cost1,1)
        pred1 = F.softmax(cost1,dim=1)
        pred1 = disparityregression(self.maxdisp)(pred1)
        cost2_ = out2
        cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost2 = torch.squeeze(cost2,1)
        pred2 = F.softmax(cost2,dim=1)
        pred2 = disparityregression(self.maxdisp)(pred2)
        next_cost = out2.clone()
        if prev_pred is not None and prev_cost is not None:
            B_prev, C_prev, D_prev, H_prev, W_prev = prev_cost.shape
            linspace = torch.linspace(start=0, end=D_prev-1, steps=D_prev, dtype=torch.float32,\
                device='cuda', requires_grad=False).view(1, D_prev, 1, 1)
            lin_disp_map = linspace.repeat(B_prev, 1, H_prev, W_prev).view(B_prev*D_prev, 1, H_prev, W_prev)
            right_of_x_BD = four_of[:, 1].unsqueeze(1).repeat(1, D_prev, 1, 1).view(B_prev*D_prev, 1, H_prev, W_prev)
            right_of_x, _ = disp_warp(right_of_x_BD, lin_disp_map, interpolate_mode= 'bilinear')
            right_of_x = right_of_x.view(B_prev, D_prev, H_prev, W_prev)
            residual_disp = right_of_x - four_of[:, 0].unsqueeze(1).repeat(1, D_prev, 1, 1)
            residual_disp = residual_disp.unsqueeze(1)
            residual_disp = residual_disp.view(B_prev, 1, D_prev, H_prev * W_prev).transpose(2,3)
            prev_cost_4d = prev_cost.view(B_prev, C_prev * D_prev, H_prev, W_prev)
            prev_cost_warped = flow_warp(prev_cost_4d, left_of, interpolate_mode='bilinear')
            prev_cost_warped = prev_cost_warped.view(B_prev, C_prev, D_prev, H_prev*W_prev).transpose(2, 3)
            refine_cost, _ = disp_warp(prev_cost_warped, residual_disp, interpolate_mode= 'bilinear')
            refine_cost = refine_cost.transpose(2, 3).view(B_prev, C_prev, D_prev, H_prev, W_prev)
            refine_cost = torch.concat((refine_cost, cost2_), dim=1)
            refine_cost = self.cost_squeeze_block(refine_cost)
            out3, _, _ = self.dres4(refine_cost, None, None)
            cost3 = self.classif3(out3)
            cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3,1)
            pred3 = F.softmax(cost3,dim=1)
            pred3 = disparityregression(self.maxdisp)(pred3)
            next_cost = out3.clone()
        debug = {"left_of": up_left_of, "right_of": up_right_of}
        if prev_gt_disp is not None:
            if curr_gt_disp is not None:
                right_of_x, _ = disp_warp(four_of[:, 1].unsqueeze(1), curr_gt_disp.unsqueeze(1), interpolate_mode= 'bilinear')
                residual_disp = right_of_x - four_of[:, 0].unsqueeze(1)
            else:
                down_pred3 = F.interpolate(pred3, scale_factor = 0.25, mode="bilinear", align_corners=False)/4.0
                right_of_x, _ = disp_warp(four_of[:, 1].unsqueeze(1), down_pred3, interpolate_mode= 'bilinear')
                residual_disp = right_of_x - four_of[:, 0].unsqueeze(1)
            nan_mask = torch.isnan(prev_gt_disp)
            prev_gt_disp[nan_mask] = 0.0
            curr_warped_disp = flow_warp(prev_gt_disp.unsqueeze(1), left_of, interpolate_mode='bilinear')
            nan_mask_warped = flow_warp(nan_mask.type(torch.float32).unsqueeze(1), left_of, interpolate_mode='bilinear') > 0
            curr_warped_disp[nan_mask_warped] = float('nan')
            curr_warped_disp = curr_warped_disp + residual_disp
        if self.training:
            return pred1, pred2, pred3, [refimg_fea, targetimg_fea], debug, curr_warped_disp, next_cost
        else:
            if prev_pred is not None and prev_cost is not None:
                output = pred3
            else:
                output = pred2
            return output, [refimg_fea, targetimg_fea], debug, next_cost
def ours_large(maxdisp, height, width, in_ch=5):
    return TESNet(maxdisp, height, width, in_ch)
