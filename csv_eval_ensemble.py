from __future__ import print_function

import numpy as np
import json
import os
import csv

import torch


def GeneralEnsemble(dets, iou_thresh = 0.5, weights=None):
    assert(type(iou_thresh) == float)
    
    ndets = len(dets)
    
    if weights is None:
        w = 1/float(ndets)
        weights = [w]*ndets
    else:
        print('len_weights',len(weights))
        print('ndets',ndets)
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



def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
    generator,
    retinanets,
    epoch,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None,
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations
    all_detections_ensemble = []
    for k in range(len(retinanets)):

        all_detections     = _get_detections(generator, retinanets[k], score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
        all_detections_ensemble.append(all_detections)

    # print('all_detections',len(all_detections))
    # print(all_detections[0])
    # print(all_detections[1])

    all_annotations    = _get_annotations(generator)


    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):

            ######## make changes to accomodate ensemble here #########
            dets = {}
            for kk in range(len(all_detections_ensemble)):
                detections     = all_detections_ensemble[kk][i][label]
                dets[kk] = detections

            ensembled_detection = None
            max_len_val = 0             # check initial value
            f = 1
            for key,val in dets.items():
                if len(val)!=1:
                    f=0              # some model predicts more than 1 bbox
            if f==0:
                for key,val in dets.items():              # if more than 1 bbox , take model which predicts max bboxs
                        max_len_val = len(val)
                        ensembled_detection = val
            else:
                bboxs_for_ensemble = []
                for kk in range(len(all_detections_ensemble)):
                    # print('dets',kk,dets[kk])
                    x1 = dets[kk][0][0]
                    y1 = dets[kk][0][1]
                    x2 = dets[kk][0][2]
                    y2 = dets[kk][0][3]
                    score = dets[kk][0][4]

                    width = x2-x1
                    height = y2-y1
                    x_c = x1+width/2
                    y_c = y1+height/2

                    bbox_for_ensemble = [x_c,y_c,width,height,int(label),score]
                    bboxs_for_ensemble.append([bbox_for_ensemble])

                # print('bboxs_for_ensemble',bboxs_for_ensemble)

                ens = GeneralEnsemble(bboxs_for_ensemble, weights = [1]*len(all_detections_ensemble))
                x1_ens,x2_ens,y1_ens,y2_ens = getCoords(ens[0])

                ensembled_detection = [[x1_ens,y1_ens,x2_ens,y2_ens,ens[0][5]]]





            # detections           = all_detections[i][label]
            detections = ensembled_detection
            # print(i,label,detections)
            annotations          = all_annotations[i][label]
            # print(annotations)

            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                # print('d',d)
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations 
    
    print('\nmAP:')
    all_maps = {}
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))
        all_maps[label_name] = str(average_precisions[label][0])
        #all_maps.append(str(average_precisions[label][0]))

    store_map_dir = './map_eval_testing/'
    if not os.path.exists(store_map_dir):
        os.makedirs(store_map_dir)

    save_name = str(epoch)+str('_eval_map.csv')
    with open(os.path.join(store_map_dir,save_name), 'w+') as writefile:
        writer = csv.writer(writefile)
        for key, value in all_maps.items():
            writer.writerow([key, value])
        print (all_maps)
    writefile.close()
    
    return average_precisions

