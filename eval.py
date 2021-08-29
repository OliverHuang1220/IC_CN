from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

os.environ["CUDA_VISIBLE_DEVICES"] = str('3')
import opts
import models
from dataloader import *
#from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="/home1/huangqiangHD/IC_CN/data/save/model-best.pth",
                    help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str, default='resnet101',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str,
                    default="/home1/huangqiangHD/IC_CN/data/save/infos_.pkl",
                    help='path to infos to evaluate')
parser.add_argument('--batch_size', type=int, default=1,
                    help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                    help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0,
                    help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                    help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=1,
                    help='Write image paths along with predictions into vis json? (1=yes,0=no)')

parser.add_argument('--sample_max', type=int, default=1,
                    help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0,
                    help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=7,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--group_size', type=int, default=1,
                    help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                    help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                    help='If 1, not allowing same word in a row')

parser.add_argument('--image_folder', type=str, default='',
                    help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='/home1/huangqiangHD/imagecaption/data/caption_images',
                    help='In case the image paths have to be preprended with a root path to an image folder')
parser.add_argument('--input_fc_dir', type=str, default='/home1/huangqiangHD/imagecaption/data/chinese_bu_fc',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='/home1/huangqiangHD/imagecaption/data/chinese_bu_att',
                    help='path to the h5file containing the preprocessed dataset')
# parser.add_argument('--input_box_dir', type=str, default='data/chinese_bu_box',
#                     help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='/home1/huangqiangHD/imagecaption/data/chinese_talk_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='/home1/huangqiangHD/imagecaption/data/chinese_talk.json',
                    help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test',
                    help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='',
                    help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='beam4_score',
                    help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=1,
                    help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0,
                    help='if we need to calculate loss.')

opt = parser.parse_args()


def convert(data):
    if isinstance(data, bytes):
        return data.decode('ascii')
    if isinstance(data, dict):
        return dict(map(convert, data.items()))
    if isinstance(data, tuple):
        return map(convert, data)
    return data


with open(opt.infos_path, 'rb') as f:
    infos = pickle.load(f)
infos = convert(infos)
s = infos['opt']
d = vars(s)
infos['opt'] = convert(d)
# Load infos


# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    #opt.input_box_dir = infos['opt'].input_box_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
for k in infos['opt'].keys():
    if k not in ignore:
        if k in opt:
            assert vars(opt)[k] == infos['opt'][k], k + ' option not consistent'
        else:
            vars(opt).update({k: infos['opt'][k]})  # copy over options from model

vocab = infos['vocab']  # ix -> word mapping
print('loading model')
# Setup the model
model = models.setup(opt)
print('a')
m = torch.load(opt.model)
print('b')
model.load_state_dict(m)
print('c')
model.cuda()
print('d')
model.eval()
print('e')
crit = utils.LanguageModelCriterion()
print('f')
# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

print('begin predict')
# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
                                                            vars(opt))

print('predict: ', split_predictions)
if lang_stats:
    print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('/home1/huangqiangHD/IC_CN/data/save/test_res7.json', 'w'))
