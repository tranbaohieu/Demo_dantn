from __future__ import absolute_import
import sys
sys.path.append('./')

import argparse
import os
import os.path as osp
import numpy as np
import math
import time
from PIL import Image, ImageFile

import torch
from torch import nn
from torch.backends import cudnn
from aster_pytorch.lib.models.model_builder import ModelBuilder
from aster_pytorch.lib.utils.serialization import load_checkpoint
from aster_pytorch.lib.evaluation_metrics.metrics import get_str_list
from aster_pytorch.lib.utils.labelmaps import get_vocabulary


class DataInfo(object):
  """
  Save the info about the dataset.
  This a code snippet from dataset.py
  """
  def __init__(self):
    super(DataInfo, self).__init__()
    # self.voc_type = voc_type

    # assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)


def create_model(resume, decoder_sdim=128, attDim=128, max_len=40, STN_ON=False, encoders =2, decoders =2):
  dataset_info = DataInfo()
  use_cuda = torch.cuda.is_available()
  if use_cuda:
    print('using cuda.')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  else:
    torch.set_default_tensor_type('torch.FloatTensor')
  model = ModelBuilder(rec_num_classes=dataset_info.rec_num_classes,
                       sDim=decoder_sdim, attDim=attDim, max_len_labels=max_len,
                       eos=dataset_info.char2id[dataset_info.EOS], STN_ON=STN_ON,
                       encoder_block = encoders, decoder_block = decoders)

  # Load from checkpoint
  if resume:
    checkpoint = load_checkpoint(resume, use_cuda)
    model.load_state_dict(checkpoint['state_dict'], strict = False)

  if use_cuda:
    device = torch.device("cuda")
    model = model.to(device)
    model = nn.DataParallel(model)

  # Evaluation
  return model.eval()

def predict(model, img, seed=1, height=32, width=100, max_len=40):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

  # Create data loaders
  if height is None or width is None:
    height, width = (32, 100)

  

  
  # img = Image.open(image_path)
  # img = process_input(img, 32, 32, 512)
  # img = image_process(image_path)
  dataset_info = DataInfo()
  use_cuda = torch.cuda.is_available()
  if use_cuda:
    with torch.no_grad():
      img = img.to(torch.device("cuda"))
  input_dict = {}
  input_dict['images'] = img.unsqueeze(0)
  # TODO: testing should be more clean.
  # to be compatible with the lmdb-based testing, need to construct some meaningless variables.
  rec_targets = torch.IntTensor(1, max_len).fill_(1)
  rec_targets[:,max_len-1] = dataset_info.char2id[dataset_info.EOS]
  input_dict['rec_targets'] = rec_targets
  input_dict['rec_lengths'] = [max_len]
  output_dict = model(input_dict)
  pred_rec = output_dict['output']['pred_rec']
  # print(output_dict['output']['pred_rec_score'])
  pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info)
  # print('Recognition result: {0}'.format(pred_str[0]))
  return pred_str[0], output_dict['output']['pred_rec_score']


if __name__ == '__main__':
  # parse the config
  args = get_args(sys.argv[1:])
  main(args)