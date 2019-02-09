
# coding: utf-8

# In[1]:


import torch
import numpy as np
import os
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
import csv_eval_ensemble as csv_eval
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader
import argparse


parser = argparse.ArgumentParser(description='Test Model')
parser.add_argument('--test_anno_file', metavar='test_anno_file', type=str,
                    help='path to csv annotation file')
parser.add_argument('--type', metavar='type', type=str,
                    help='improved if single model, improved_ensemble for ensemble')

args = parser.parse_args()

# In[2]:


def GeneralEnsemble(dets, iou_thresh = 0.5, weights=None):
    assert(type(iou_thresh) == float)
    
    ndets = len(dets)
    
    if weights is None:
        w = 1/float(ndets)
        weights = [w]*ndets
    else:
        assert(len(weights) == ndets)
        
        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()
    
    for idet in range(0,ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue
                
            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]
                
                if odet == det:
                    continue
                
                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = computeIOU(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox
                                
                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox,w))
                    used.append(bestbox)
                            
            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)
                
                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0
                
                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w*b[0]
                    yc += w*b[1]
                    bw += w*b[2]
                    bh += w*b[3]
                    conf += w*b[5]
                
                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum    

                new_box = [xc, yc, bw, bh, box[4], conf]
                out.append(new_box)
    return out
    


# In[3]:


def getCoords(box):
    x1 = float(box[0]) - float(box[2])/2
    x2 = float(box[0]) + float(box[2])/2
    y1 = float(box[1]) - float(box[3])/2
    y2 = float(box[1]) + float(box[3])/2
    return x1, x2, y1, y2


# In[4]:


def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)
    
    x_left   = max(x11, x21)
    y_top    = max(y11, y21)
    x_right  = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0    
        
    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)        
    
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou


# In[5]:


if __name__=="__main__":
    # Toy example
    # dets = [ 
    #         [[0.1, 0.1, 1.0, 1.0, 0, 0.9], [1.2, 1.4, 0.5, 1.5, 0, 0.9]],
    #         [[0.2, 0.1, 0.9, 1.1, 0, 0.8]],
    #         [[5.0,5.0,1.0,1.0,0,0.5]]
    #        ]
    
    # ens = GeneralEnsemble(dets, weights = [1.0, 0.1, 0.5])
    # print(ens)


# In[6]:


    device = torch.device('cuda')


    # ### Set path to model weights

    # In[7]:


    my_models = []
    if args.type=='improved':
        model_wt_path = './Weighted_Single/csv_retinanet_19.pt'
        model = torch.load(model_wt_path)
        model = model.to(device)
        model.eval()
        my_models.append(model)

    elif args.type=='improved_ensemble':
        model_wt_path1 = './Weighted_Ensemble/csv_retinanet_19.pt'
        model1 = torch.load(model_wt_path1)
        model1 = model1.to(device)
        model1.eval()
        my_models.append(model1)

        model_wt_path2 = './Weighted_Ensemble/csv_retinanet_14.pt'
        model2 = torch.load(model_wt_path2)
        model2 = model2.to(device)
        model2.eval()
        my_models.append(model2)




    test_file_path = args.test_anno_file
    # test_file_path = 'my_test.csv'

    csv_classes_path = 'classname2id.csv'
    epoch_num = 0
    dataset_test = CSVDataset(train_file=test_file_path, class_list=csv_classes_path, transform=transforms.Compose([Normalizer(), Resizer()]))
    mAP = csv_eval.evaluate(dataset_test, my_models, epoch_num)
    # print(mAP)
    print(mAP)
    print('mAP over all classes', np.mean(list(mAP.values())))


    # In[ ]:


    # get_ipython().system(u'pwd')


# In[ ]:




