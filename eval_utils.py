from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# i
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils


def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("/home1/huangqiangHD/IC_CN/AI_Challenger/Evaluation/caption_eval")
    annFile = '/home1/huangqiangHD/imagecaption/data/eval_reference.json'
    from coco_caption.pycxtools.coco import COCO
    from coco_caption.pycxevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('/home1/huangqiangHD/IC_CN/data/eval_result'):
        os.mkdir('/home1/huangqiangHD/IC_CN/data/eval_result')
    cache_path = os.path.join('/home1/huangqiangHD/IC_CN/data/eval_result/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()
    # filter results to only those in MSCOCO validation set (will be about a third)
    # preds_filt = [p for p in preds if p['image_id'] in valids]
    # print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    print(len(set(cocoRes.getImgIds()) & set(coco.getImgIds())))
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    # for p in preds:
    #     image_id, caption = p['image_id'], p['caption']
    #     imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 7)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:]).data[0]
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        # data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        # fc_feats, att_feats, att_masks = tmp
        fc_feats, att_feats = tmp
        # forward the model to also get generated samples for each image
        seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)
        seq = seq.cpu().numpy()

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join(
                    [utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            if verbose:
                # print('image %s: ' %(data['infos'][k]['id']), sent.encode('utf8', 'replace'))
                print('image %s: ' % (data['infos'][k]['id']), sent)
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                            data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
                    len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats
