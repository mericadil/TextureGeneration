# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags

# ------------------------------modify this to your own data path--------------------------------------------
# path of pretrained re-id weight network
flags.DEFINE_string('reid_weight_path',
                    '/home/wangj/Models/TextureGeneration/reid_models/checkpoint_120.pth.tar',
                    'weight path for reid')

flags.DEFINE_string('market1501_dir',
                    '/home/wangj/Datasets/market1501',
                    'directory of market1501 dataset')

flags.DEFINE_string('surreal_texture_path',
                    '/home/wangj/Datasets/SURREAL/smpl_data/textures',
                    'surreal texture dataset')

flags.DEFINE_string('CUHK_SYSU_path',
                    '/home/wangj/Datasets/CUHK-SYSU',
                    'CUHK SYSU dataset')

flags.DEFINE_string('PRW_img_path',
                    '/home/wangj/Datasets/PRW/frames',
                    'prw dataset raw frame path')

flags.DEFINE_string('market1501_render_tensor_dir',
                    '/home/wangj/Datasets/Texture/market1501_rendering_matrix_new',
                    'directory of rendering tensor of market1501')

# -----------------------finish setting dataset path---------------------------------------------------------

# -----------------------Start Setting Model Logging Path------------------------------------------------
flags.DEFINE_string('model_log_path', '/home/wangj/Models/TextureGeneration/model_log',
                    'model save path')
flags.DEFINE_string('runs_log_path', '/home/wangj/Models/TextureGeneration/runs_log',
                    'run log save path')
# -----------------------Finish Setting Model Logging Path-----------------------------------


# ---------------------------training parameters-------------------------------------------------------------
flags.DEFINE_integer('num_instance', 4, 'num_instance')
flags.DEFINE_integer('epoch', 120, 'train epoch num')
flags.DEFINE_integer('batch_size', 16, 'Input batch size after pre-processing')
flags.DEFINE_float('learning_rate', 1e-4, 'generator learning rate')
flags.DEFINE_float('weight_decay', 1e-5, 'weight decay')
flags.DEFINE_integer('log_step', 2000, 'log step')
flags.DEFINE_integer('runs_log_step', 10, 'runs log step')
flags.DEFINE_integer('eval_step', 10000, 'eval step')
flags.DEFINE_integer('worker_num', 4, 'number of data loader workers')

flags.DEFINE_integer('gpu_nums', 1, 'gpu ids')

flags.DEFINE_string('pretrained_model_path', None, "use the pre_trained model on the generated data to do fine tune")

flags.DEFINE_string('log_name', '', 'define the log name, convenient for recognizing the model and run log')

flags.DEFINE_string('model', 'unet', 'use which model')

flags.DEFINE_integer('num_classes', 86642, 'num of classes of reid model')

flags.DEFINE_string('reid_model', 'market1501', 'use which reid model')

# loss weights
flags.DEFINE_float('reid_triplet_loss_weight', 0, 'weight of triplet feature reid loss')
flags.DEFINE_float('reid_softmax_loss_weight', 0, 'weight of softmax feature reid loss')
flags.DEFINE_float('face_loss_weight', 1.0, 'weight of face loss')
flags.DEFINE_float('perceptual_loss_weight', 5000, 'weight of perceptual loss')
flags.DEFINE_float('reid_triplet_hard_loss_weight', 0.0, 'weight of triplet hard reid loss')
flags.DEFINE_float('reid_triplet_loss_not_feature_weight', 0, 'weight of triplet reid loss')
flags.DEFINE_float('uvmap_intern_loss_weight', 0, 'weight of uvmap intern loss')
flags.DEFINE_float('fake_and_true_loss_weight', 0, 'weight of fake and true loss')

flags.DEFINE_float('margin', 0.3, 'margin for triplet hard loss')

flags.DEFINE_integer('texture_size', 64, 'size of generated texture')

flags.DEFINE_integer('epoch_now', 0, 'epoch start num')

flags.DEFINE_integer('layer', 5, 'which layer\'s feature')

flags.DEFINE_integer('triplet', 1, 'use triplet or not')

flags.DEFINE_bool('use_real_background', True, 'whether use real background or no background')

#Newly added for texformer comparison
ra_body_path = '/auto/k2/adundar/3DSynthesis/data/texformer/meta/ra_body.pkl'
VERTEX_TEXTURE_FILE = '/auto/k2/adundar/3DSynthesis/data/texformer/meta/vertex_texture.npy'
cube_parts_path = '/auto/k2/adundar/3DSynthesis/data/texformer/meta/cube_parts_12.npy'
# -------------------------------------finish training parameters----------------------------------------

SMPL_OBJ = 'smpl/models/body.obj'
SMPL_MODEL = 'smpl/models/neutral.pkl'
IMG_SIZE = 224

TRANS_MAX = 20  # value of jitter translation
SCALE_MAX = 1.23  # Max value of scale jitter
SCALE_MIN = 0.8  # Min value of scale jitter
INPUT_DIM = 3  # input dim, always 3
OUTPUT_DIM = 3  # output dim, always 3

# define train super parameters
flags.DEFINE_integer('h', 128, 'image height')
flags.DEFINE_integer('w', 64, 'image width')
flags.DEFINE_integer('z_size', 256, 'size of random z')


def get_config():
    config = flags.FLAGS

    config(sys.argv)

    return config


if __name__ == '__main__':
    config = get_config()
    print(config.worker_num)
