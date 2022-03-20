from NMR.neural_render_test import NrTextureRenderer
import torch
import cv2
import argparse
import numpy as np
import os
import pickle


class Demo:
    def __init__(self, model_path):
        print(model_path)

        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()

    def generate_texture(self, img_path):
        img = cv2.imread(img_path)


        img = cv2.resize(img, (64, 128))
        img = (img / 225. - 0.5) * 2.0
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)

        out = self.model(img)

        out = out.cpu().detach().numpy()[0]
        out = out.transpose((1, 2, 0))
        out = (out / 2.0 + 0.5) * 255.
        out = out.astype(np.uint8)
        out = cv2.resize(out, dsize=(64, 64))

        #changed again to feed renderer
        out = out.transpose((2, 0, 1))
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show generated image')
    parser.add_argument('--gpu', '-g')
    parser.add_argument('--img', '-i')
    parser.add_argument('--model', '-m', default='model_path')
    parser.add_argument('--out', '-o', default=None)

    #add smpl_dir to read pickle file for verts and cam params
    parser.add_argument('--dir', '-d', default='/auto/k2/adundar/3DSynthesis/data/texformer/datasets/SMPLMarket')

    args = parser.parse_args()
    img_path = args.img
    out_path = args.out
    model_path = args.model
    smpl_data_dir = args.dir
    renderer = NrTextureRenderer(render_res=128, device='cuda:0')

    torch.nn.Module.dump_patches = True
    
    demo = Demo(model_path)
    smpl_dir = os.path.join(smpl_data_dir, 'SMPL_RSC', 'pkl')

    print(img_path)
    for root, dir, names in os.walk(img_path):
        for name in names:
            full_path = os.path.join(img_path, name)
            print('executing: ', full_path)
            uvmap = torch.from_numpy(demo.generate_texture(img_path=full_path)).to('cuda:0').float()
            
            #Add batch size
            uvmap = torch.unsqueeze(uvmap, 0)

            print("*********************************************")
            print(uvmap.shape)
            print('finish: ', os.path.join(out_path, name))

            pkl_path = os.path.join(smpl_dir, name[:-4]+'.pkl')
            print(pkl_path)

            with open(pkl_path, 'rb') as f:
                smpl_list = pickle.load(f)

            verts = torch.from_numpy(smpl_list[0])
            
            verts = verts.view(1, -1, 3)
            
            #Verts dimension debugging
            print(verts)
            print(verts.shape)
            verts = verts.to('cuda:0')
            print(verts.ndimension())

            cam_t = torch.from_numpy(smpl_list[1])
            cam_t = torch.unsqueeze(cam_t, 0)
            cam_t = cam_t.to('cuda:0')
            
            rendered_img, depth, mask = renderer.render(verts, cam_t, uvmap)
            rendered_img = rendered_img.squeeze(0).cpu().numpy()
            rendered_img = rendered_img.transpose((1, 2, 0))

            print(rendered_img.shape)
            
            cv2.imwrite(os.path.join(out_path, name), rendered_img)
