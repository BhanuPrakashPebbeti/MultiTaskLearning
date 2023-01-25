import torch as T
from torch.nn import functional as F
from torch import nn
import cv2
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
import matplotlib.pyplot as plt
import timm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import transforms
import torchvision.transforms.functional as tvf
import numpy as np
import os
import time
import math
from prettytable import PrettyTable
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from tensordash.torchdash import Torchdash
# import sys
# sys.path.append(".")
# from pcgrad import PCGrad

device = 'cuda'

histories_val = Torchdash(
    ModelName = 'MTL VAL',
    email = 'pebbetibhanu2017@gmail.com', 
    password = 'Flight$5909')

seg_histories_val = Torchdash(
    ModelName = 'MTL SEG VAL',
    email = 'pebbetibhanu2017@gmail.com', 
    password = 'Flight$5909')

dep_histories_val = Torchdash(
    ModelName = 'MTL DEPTH VAL',
    email = 'pebbetibhanu2017@gmail.com', 
    password = 'Flight$5909')

det_histories_val = Torchdash(
    ModelName = 'MTL DETECT VAL',
    email = 'pebbetibhanu2017@gmail.com', 
    password = 'Flight$5909')

class Encoder(nn.Module):
    def __init__(self, backbone = 'resnet18', device = 'cuda'):
        super(Encoder, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained = True)
        self.List = list(self.backbone.children())[:-2]
        self.device = device
    def forward(self,X):
        outputs = []
        X = X.to(self.device).float()
        for i,layer in enumerate(self.List):
            X = layer(X)
            if i>1:
                outputs.append(X)
        return outputs
 
class objdet_Decoder(nn.Module):
    '''series of convs ==> final output heatmap'''
    def __init__(self, n_classes, stride = 2, device = 'cuda'):
        super(objdet_Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear')
        self.conv1 = nn.Conv2d(512,256,(3,3),padding = 1)  # 16
        self.conv2 = nn.Conv2d(256,128,(3,3),padding = 1)  #32
        self.conv3 = nn.Conv2d(128,64,(3,3),padding = 1) #64
        self.conv4 = nn.Conv2d(64,32,(3,3),padding = 1) #128
        self.hmap = nn.Conv2d(32,n_classes,(1,1)) #128
        self.regs = nn.Conv2d(32,2,(1,1))
        self.w_h_ = nn.Conv2d(32,2,(1,1))
        
    def forward(self,X):
        X = self.upsample(X[-1])
        X = F.relu(self.conv1(X))
        X = self.upsample(X)
        X = F.relu(self.conv2(X))
        X = self.upsample(X)
        X = F.relu(self.conv3(X))
        X = self.upsample(X)
        X = F.relu(self.conv4(X))
        return [[T.sigmoid(self.hmap(X)), T.sigmoid(self.regs(X)), T.sigmoid(self.w_h_(X))]]
        
        
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)
        
class up(nn.Module):
    '''down samling--->double conv'''
    def __init__(self,in_channels, out_channels,last_layer=False):
        super(up,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear')
        if last_layer:
            self.conv = DoubleConv(in_channels*2,out_channels)
        else:
            self.conv = DoubleConv(in_channels*3//2,out_channels)   #since we are concatenating 
    def forward(self,x1,x2):
        x1 = self.upsample(x1)
        X = T.cat([x1,x2],dim=1)
        X = self.conv(X)
        return X
        
class seg_decoder(nn.Module):
    def __init__(self, n_classes = 23, device="cuda"):
        super(seg_decoder, self).__init__()
        
        self.up1 = up(512,256)
        self.up2 = up(256,128)
        self.up3 = up(128,64)
        self.up4 = up(64,32,last_layer=True)
        self.out_conv = nn.Conv2d(32,n_classes,(3,3),padding=1)
    
    def forward(self,outputs):
        X = self.up1(outputs[-1],outputs[-2])
        X = self.up2(X,outputs[-3])
        X = self.up3(X,outputs[-4])
        X = self.up4(X,outputs[-6])
        X = self.out_conv(X)
        return X
     
class MTL_Model(nn.Module):
    def __init__(self,n_classes = 35,device='cuda'):
        super(MTL_Model,self).__init__()
        self.encoder = Encoder(device=device)
        self.seg_decoder = seg_decoder(n_classes ,device=device)
        self.dep_decoder = seg_decoder(n_classes = 1,device=device)
        self.obj_decoder = objdet_Decoder(n_classes = 15,device=device)
        self.to(device)
        
    def forward(self,X):
        outputs = self.encoder(X)
        seg_maps = self.seg_decoder(outputs)
        depth_maps = self.dep_decoder(outputs)
        detection_maps = self.obj_decoder(outputs)
        return (seg_maps, T.sigmoid(depth_maps),detection_maps)

PALETTE = {
    (128, 64,128)  : 0 , #'road' 
    (250,170,160) : 1 , #'parking'  
    ( 81,  0, 81) : 2 ,#drivable fallback
    (244, 35,232) : 3 , #sidewalk
    (230,150,140) : 4 , #rail track
    (152,251,152) : 5 ,#non-drivable fallback
    (220, 20, 60) : 6 ,#person
    (246, 198, 145) : 7 ,#animal
    (255,  0,  0) : 8 , #rider
    (  0,  0,230) : 9 ,#motorcycle
    (119, 11, 32) : 10 ,  #bicycle
    (255, 204, 54) : 11,#autorickshaw
    (  0,  0,142) : 12,  #car
    (  0,  0, 70) : 13, #truck
    (  0, 60,100) : 14,    #bus
    (  0,  0, 90) : 15,#caravan
    (  0,  0,110) : 16,#trailer
    (  0, 80,100) : 17,#train
    (136, 143, 153) : 18,#vehicle fallback
    (220, 190, 40) : 19,#curb
    (102,102,156) : 20,#wall
    (190,153,153) : 21,#fence
    (180,165,180) : 22,#guard rail
    (174, 64, 67) : 23,#billboard
    (220,220,  0) : 24,#traffic sign
    (250,170, 30) : 25,#traffic light
    (153,153,153) : 26,#pole
    (169, 187, 214) : 27,#obs-str-bar-fallback
    ( 70, 70, 70) : 28,#building
    (150,100,100) : 29,#bridge
    (150,120, 90) : 30,#tunnel
    (107,142, 35) : 31,#vegetation
    ( 70,130,180) : 32,#sky
    (169, 187, 214) : 33,#fallback background
    (  0,  0,  0) : 34#unlabeled
}

def convert_from_color_segmentation(arr_3d):
    arr_3d = np.array(arr_3d)
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    palette = PALETTE
    for i in range(0, arr_3d.shape[0]):
        for j in range(0, arr_3d.shape[1]):
            key = (arr_3d[i, j, 2], arr_3d[i, j, 1], arr_3d[i, j, 0])
            arr_2d[i, j] = palette.get(key,34) # default value if key was not found is 0

    return arr_2d

def labels_to_cityscapes_palette(array):
    result = np.zeros((array.shape[0], array.shape[1], 3))
    for value, key in PALETTE.items():
        result[np.where(array == key)] = (value[2],value[1],value[0])
    return result/255

def to_one_hot(mask, n_classes=35):
    one_hot = np.zeros((mask.shape[0], mask.shape[1], n_classes))
    for i, unique_value in enumerate(np.unique(mask)):
        one_hot[:, :, unique_value][mask == unique_value] = 1
    return one_hot    

class FocalLoss(nn.Module):
    def __init__(self, alpha=4, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = T.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return T.mean(F_loss)
        else:
            return F_loss
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, eps=1e-7):
        inputs = F.softmax(inputs, dim=1)
        targets = targets.type(inputs.type())
        intersection = T.sum(inputs * targets, (0, 2, 3))
        cardinality = T.sum(inputs + targets, (0, 2, 3))
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)
    
class DiceFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceFocalLoss, self).__init__()
        self.criterion = FocalLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, inputs, targets):
        targets_ = T.argmax(targets, dim=1)
        floss = self.criterion(inputs, targets_.long())
        dice_loss = self.dice_loss(inputs,targets)
        Dice_BCE = floss + dice_loss
        return Dice_BCE

class DepthLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DepthLoss, self).__init__()
    def im_gradient_loss(self,d_batch, n_pixels):
        a = T.Tensor([[[[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]]]])
                      
        b = T.Tensor([[[[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]]]])
        
        a = a.to(device)
        b = b.to(device)

        G_x = F.conv2d(d_batch, a, padding=1).to(device)
        G_y = F.conv2d(d_batch, b, padding=1).to(device)
        
        G = T.pow(G_x,2)+ T.pow(G_y,2)
    
        return G.view(-1, n_pixels).mean(dim=1).mean()

    def forward(self,preds, actual_depth):
        
        n_pixels = actual_depth.shape[2]*actual_depth.shape[3]
        preds = preds*1000
        preds[preds<=0] = 0.00001
        actual_depth[actual_depth==0] = 0.00001
        d = T.log(preds) - T.log(actual_depth)
        grad_loss_term = self.im_gradient_loss(d, n_pixels)
        term_1 = T.pow(d.view(-1, n_pixels),2).mean(dim=1).mean() #pixel wise mean, then batch sum
        term_2 = (T.pow(d.view(-1, n_pixels).sum(dim=1),2)/(2*(n_pixels**2))).mean()
        loss1 = term_1 - term_2 + grad_loss_term
        loss2 = F.mse_loss(preds,actual_depth,reduction='mean')
        return loss1 + loss2
    
class DetectionLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DetectionLoss, self).__init__()
        
    def forward(self,obj, hmap, regs, w_h_):
        regs = [self._tranpose_and_gather_feature(r, obj['inds']) for r in regs]
        w_h_ = [self._tranpose_and_gather_feature(r, obj['inds']) for r in w_h_]
        hmap_loss = self._neg_loss(hmap, obj['hmap'])
        reg_loss = self._reg_loss(regs, obj['regs'], obj['ind_masks'])
        w_h_loss = self._reg_loss(w_h_, obj['w_h_'], obj['ind_masks'])
        loss =  0.5*hmap_loss +  reg_loss +  w_h_loss 
        return loss 
    
    def _neg_loss(self,preds, targets):
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()
        neg_weights = T.pow(1 - targets, 4)
        loss = 0
        for pred in preds:
            pred = T.clamp(T.sigmoid(pred), min=1e-4, max=1 - 1e-4)
            pos_loss = T.log(pred) * T.pow(1 - pred, 2) * pos_inds
            neg_loss = T.log(1 - pred) * T.pow(pred, 2) * neg_weights * neg_inds

            num_pos = pos_inds.float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if num_pos == 0:
                loss = loss - neg_loss
            else:
                loss = loss - (pos_loss + neg_loss) / num_pos
        return loss / len(preds)
    
    def _reg_loss(self,regs, gt_regs, mask):
        mask = mask[:, :, None].expand_as(gt_regs).float()
        loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
        return loss / len(regs)
    
    def _gather_feature(self,feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat
    
    def _tranpose_and_gather_feature(self,feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feature(feat, ind)
        return feat

    
def _neg_loss(preds, targets):
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = T.pow(1 - targets, 4)

    loss = 0
    for pred in preds:
        pred = T.clamp(pred, min=1e-4, max=1 - 1e-4)
        pos_loss = T.log(pred) * T.pow(1 - pred, 2) * pos_inds
        neg_loss = T.log(1 - pred) * T.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)


def _reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)

input_size_x,input_size_y = (640, 480)
MODEL_SCALE = 2

def _gather_feature(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feature(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feature(feat, ind)
    return feat

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def convert(obj,width,height):
    x_scale = 640 / width
    y_scale = 480 / height
    x_c = int(np.round(((obj[0]+obj[2])/2)*x_scale))
    y_c = int(np.round(((obj[1]+obj[3])/2)*y_scale))
    w = int(np.round((obj[2]-obj[0])*x_scale))
    h = int(np.round((obj[3]-obj[1])*y_scale))
    box = [x_c,y_c,w,h]
    return box

def make_hm_regr(target,width,height,num_classes = 15,input_size_x = 640,input_size_y = 480,MODEL_SCALE=2,max_objs=240,gaussian_iou = 0.7):
    hmap = np.zeros((num_classes, input_size_y//MODEL_SCALE, input_size_x//MODEL_SCALE), dtype=np.float32)
    w_h_ = np.zeros((max_objs, 2), dtype=np.float32)
    regs = np.zeros((max_objs, 2), dtype=np.float32)
    inds = np.zeros((max_objs,), dtype=np.int64)
    ind_masks = np.zeros((max_objs,), dtype=np.uint8)
    boxes = literal_eval(target["bbox"])
    classes = {"bicycle":0,"bus":1,"traffic sign":2,"train":3,"motorcycle":4,"car":5,"traffic light":6,"person":7,"vehicle fallback":8,"truck":9,"autorickshaw":10,"animal":11,"caravan":12,"rider":13,"trailer":14}

    for i,a in enumerate(boxes):
        box_ = a["bbox"]
        box = convert(box_,width,height)
        if (box[0]>640) or (box[1]>480):
            continue
        center = np.array([(box[0]),(box[1])], dtype=np.float32)
        obj_c = np.array([(box[0]//MODEL_SCALE),(box[1]//MODEL_SCALE)], dtype=np.float32)
        obj_c_int = obj_c.astype(np.int32)
        h = box[3]
        w = box[2]
        if h > 0 and w > 0:
            radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), gaussian_iou)))
            hmap[classes[a["label"]],:,:] = draw_umich_gaussian(hmap[classes[a["label"]],:,:], obj_c_int, radius)   
            w_h_[i] =  w/input_size_x, h/input_size_y
            regs[i] = center - (obj_c_int*MODEL_SCALE)
            inds[i] = ((obj_c_int[1]) * (input_size_x//MODEL_SCALE)) + (obj_c_int[0])
            ind_masks[i] = 1
    return {'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks}

class MTL(Dataset):
    def __init__(self, filename=None, input_size=(640, 480), output_size=(320, 240), n_classes=15):
        super().__init__()
        self.filename = filename
        self.n_classes = n_classes
        self.max_objs = 240
        self.gaussian_iou = 0.7
        self.dataset = pd.read_csv(self.filename)
        self.input_size = input_size
        self.output_size = output_size
        self.input_size_x = self.input_size[0]
        self.input_size_y = self.input_size[1]
        self.MODEL_SCALE = self.input_size[0]//self.output_size[0]
        self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.resize1 = transforms.Compose([transforms.Resize(self.input_size)])
        self.resize2 = transforms.Compose([transforms.Resize(self.output_size)])


    def __len__(self): return len(self.dataset)
    
    def __getitem_internal__(self, idx, preprocess=True):
        target = self.dataset.iloc[idx]
        rgb_image = cv2.imread(target["Path"])
        height, width, channels = rgb_image.shape
        rgb_image = cv2.resize(rgb_image,self.input_size)
        obj = make_hm_regr(target,width,height,self.n_classes,self.input_size_x,self.input_size_y,self.MODEL_SCALE,self.max_objs,self.gaussian_iou)
        seg_mask = np.load(target["Seg_Path"])
        depth_image = np.load(target["Depth_path"])
        depth_image = cv2.resize(depth_image,self.output_size)
        seg_mask = cv2.resize(seg_mask,self.output_size)
        one_hot_segmask = to_one_hot(seg_mask)
        if preprocess:
            rgb_image = self.preprocess(np.array(rgb_image))
            one_hot_segmask = transforms.ToTensor()(np.array(one_hot_segmask))
            depth_image = transforms.ToTensor()(np.array(depth_image))
        else:
            rgb_image = transforms.ToTensor()(np.array(rgb_image))
            one_hot_segmask = transforms.ToTensor()(np.array(one_hot_segmask))
            depth_image = transforms.ToTensor()(np.array(depth_image))
            seg_mask = transforms.ToTensor()(np.array(seg_mask))
        return (rgb_image,seg_mask,one_hot_segmask,depth_image, obj)

    def __getitem__(self, idx):
        return self.__getitem_internal__(idx, True)
    
    def raw(self, idx):
        return self.__getitem_internal__(idx, False)
    
model = MTL_Model(device = device)
print(device)

train_dataloader = MTL("/home/b170007ec/Programs/MTL/DSD_MTL/Dataset/train_dataset.csv")
print("Train :",train_dataloader.__len__())
val_dataloader = MTL("/home/b170007ec/Programs/MTL/DSD_MTL/Dataset/val_dataset.csv")
print("Val :",val_dataloader.__len__())

diceloss = DiceFocalLoss()
depthloss = DepthLoss()

def loss_fn(y_pred, y_true, obj, hmap, regs, w_h_):
    (pred_seg, pred_depth) = y_pred
    (true_seg, true_depth) = y_true
    dice = diceloss(pred_seg, true_seg)
    depth = depthloss(pred_depth, true_depth)
    #detect = detectionloss(obj, hmap, regs, w_h_)
    regs = [_tranpose_and_gather_feature(r, obj['inds']) for r in regs]
    w_h_ = [_tranpose_and_gather_feature(r, obj['inds']) for r in w_h_]
    hmap_loss = _neg_loss(hmap, obj['hmap'])
    reg_loss = _reg_loss(regs, obj['regs'], obj['ind_masks'])
    w_h_loss = _reg_loss(w_h_, obj['w_h_'], obj['ind_masks'])
    detect =  0.5*hmap_loss +  reg_loss +  w_h_loss 
    return dice+depth+detect, dice, depth, detect 

@T.no_grad()
def validation(model, loader, loss_fn):
    vlosses = []
    dice_vloss = []
    depth_vloss = []
    detect_vloss = []
    model.eval()
    for rgb,seg_mask,seg,depth,obj in loader:
        rgb,seg,depth = rgb.to(device), seg.to(device), depth.to(device)
        obj['hmap'], obj['w_h_'], obj['regs'], obj['inds'], obj['ind_masks'] = obj['hmap'].to(device), obj['w_h_'].to(device), obj['regs'].to(device), obj['inds'].to(device), obj['ind_masks'].to(device)
        y_pred = model(rgb)
        hmap, regs, w_h_ = zip(*y_pred[2])
        y_true = (seg,depth)
        loss, v_dice, v_depth, v_detect = loss_fn((y_pred[0],y_pred[1]), y_true, obj, hmap, regs, w_h_)
        dice_vloss.append(v_dice.item())
        depth_vloss.append(v_depth.item())
        detect_vloss.append(v_detect.item())
        vlosses.append(loss.item())
    return np.array(vlosses).mean(), np.array(dice_vloss).mean(), np.array(depth_vloss).mean(), np.array(detect_vloss).mean()

def send_msg(epoch,total_epochs,loss_v,segvloss,dpvloss,dtvloss):
    histories_val.sendLoss(loss = loss_v, epoch = epoch, total_epochs = total_epochs)
    seg_histories_val.sendLoss(loss = segvloss, epoch = epoch, total_epochs = total_epochs)
    dep_histories_val.sendLoss(loss = dpvloss, epoch = epoch, total_epochs = total_epochs)
    det_histories_val.sendLoss(loss = dtvloss, epoch = epoch, total_epochs = total_epochs)

batch_size = 50
EPOCHES = 250

train_loader = DataLoader(train_dataloader,batch_size=batch_size,shuffle=False, num_workers=0, sampler=SubsetRandomSampler(list(range(train_dataloader.__len__()))),
                             drop_last=False)
val_loader = DataLoader(val_dataloader,batch_size=batch_size,shuffle=False,
                              num_workers=0,
                              sampler=SubsetRandomSampler(list(range(len(val_dataloader.dataset)))),
                             drop_last=False)
raw_line0 = r'''Epoch[{}]    |    Lr:{}'''
raw_line1 = r'''Train Loss:[SEG:{}+DEPTH:{}+DETECT:{}] | Val Loss:[SEG:{}+DEPTH:{}+DETECT:{}]'''
raw_line3 = r'''TOTAL Train loss: {}  |  TOTAL Val loss: {}  |  Time:{:.1f} min '''
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

optimizer = T.optim.Adam([
                {'params': model.parameters()}]
                , lr=0.00001)
# optimizer_pc = PCGrad(optimizer)

scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1,patience=2,verbose=True)
best_loss = None

model.load_state_dict(T.load("/home/b170007ec/Programs/MTL/DSD_MTL/MTL_V3/model_v3-1.8697537092062144.pth",map_location=T.device('cuda')))

try:
    for epoch in range(1, EPOCHES+1):
        losses = []
        dice_loss = []
        depth_loss = []
        detect_loss = []
        start_time = time.time()
        t = tqdm(train_loader)
        model.train()
        for i,(rgb,seg_mask,seg,depth,obj) in enumerate(t):
            rgb,seg,depth = rgb.to(device), seg.to(device), depth.to(device)
            obj['hmap'], obj['w_h_'], obj['regs'], obj['inds'], obj['ind_masks'] = obj['hmap'].to(device), obj['w_h_'].to(device), obj['regs'].to(device), obj['inds'].to(device), obj['ind_masks'].to(device)
            optimizer.zero_grad()
            y_pred = model(rgb)
            hmap, regs, w_h_ = zip(*y_pred[2])
            loss, dice, depth, detect = loss_fn((y_pred[0],y_pred[1]), (seg,depth), obj, hmap, regs, w_h_)
#             losses = [dice,depth,detect]
            loss.backward()
            optimizer.step()
            dice_loss.append(dice.item())
            depth_loss.append(depth.item())
            detect_loss.append(detect.item())
            losses.append(loss.item())
        vloss, vdice, vdepth, vdetect = validation(model, val_loader, loss_fn)
        send_msg(epoch,EPOCHES,vloss, vdice, vdepth, vdetect)
        print(raw_line0.format(epoch,optimizer.param_groups[0]["lr"]))
        print(raw_line1.format(np.array(dice_loss).mean(),np.array(depth_loss).mean(),np.array(detect_loss).mean(),vdice,vdepth,vdetect))
        print(raw_line3.format(np.array(losses).mean(),vloss,(time.time()-start_time)/60**1))

        if best_loss == None:
            best_loss = vloss
            T.save(model.state_dict(), '/home/b170007ec/Programs/MTL/DSD_MTL/Models/model_v3-{}.pth'.format(best_loss))
            print("saving model ..")
        if vloss < best_loss:
            best_loss = vloss
            T.save(model.state_dict(), '/home/b170007ec/Programs/MTL/DSD_MTL/Models/model_v3-{}.pth'.format(best_loss))
            print("saving model ..")
        scheduler.step(vloss)
except:
    histories_val.sendCrash()