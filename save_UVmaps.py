from NMR.neural_render_test import NrTextureRenderer
import torch
import cv2
import argparse
import numpy as np
import os.path as osp
import pickle
from tqdm import tqdm

class Saver:
    def __init__(self, model_path, data_dir, output_path):
        print(model_path)

        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.output_path = output_path
        self.data_dir = data_dir

        paths_pkl_path = osp.join(self.data_dir, 'eval_list.pkl')
        with open(paths_pkl_path, 'rb') as f:
            self.img_paths = pickle.load(f)

    def generate_texture(self, img_path):
        img = cv2.imread(osp.join(self.data_dir, img_path))


        img = cv2.resize(img, (64, 128))
        img = (img / 225. - 0.5) * 2.0
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)

        out = self.model(img)

        out = out.cpu().detach().numpy()[0]
        out = out.transpose((1, 2, 0))
        out = (out / 2.0 + 0.5) * 255.
        out = out.astype(np.uint8)
        out = cv2.resize(out, dsize=(64, 64))
        
        return out
    
    def save_all_UV_maps(self):
        print(len(self.img_paths))
        for img_path in tqdm(self.img_paths): 
            image = self.generate_texture(img_path)
            img_name = img_path.split('/')[-1]
            cv2.imwrite(osp.join(self.output_path, img_name), image)




if __name__ == '__main__':
    #add smpl_dir to read pickle file for verts and cam params

    model_path = 'pretrained_model/pretrained_weight.pkl'
    smpl_data_dir = '/auto/k2/adundar/3DSynthesis/data/texformer/datasets/SMPLMarket'
    output_path = '/auto/k2/adundar/3DSynthesis/data/texformer/datasets/TextureGenerationResults'
    torch.nn.Module.dump_patches = True
    
    demo = Saver(model_path, smpl_data_dir, output_path)
    demo.save_all_UV_maps()