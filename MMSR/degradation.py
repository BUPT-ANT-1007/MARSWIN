import numpy as np
import random
import torch
from torch.nn import functional as F
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG, USMSharp
import cv2
import yaml

import math
import os
import os.path as osp
import time
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor,tensor2img
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from perlin_cupy import generate_fractal_noise_2d


@DATASET_REGISTRY.register()
class MMSRDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(MMSRDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(' ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d

    def __len__(self):
        return len(self.paths)


class ImageDegradationModule:
    def __init__(self, opt):
        self.opt = opt
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.kernel1 = opt['kernel1'].to('cuda')
        self.kernel2 = opt['kernel2'].to('cuda')
        self.sinc_kernel = opt['sinc_kernel'].to('cuda')
        self.gt = opt['gt'].to('cuda')
    
    def degrade_image(self,image):
        
        # generate dusty
        fai = np.array([0.26131062, 0.4419994, 0.68503087]) #经过统计dusty patch 计算得到的值
        #生成噪声
        np.random.seed(0)
        noise = np.zeros([3072,4096])
        dusty_type = random.choice(['type1', 'type2', 'type3'])
        if dusty_type == 'type1':
            noise = generate_fractal_noise_2d((3072, 4096), (1, 1), octaves=8, persistence=0.5, lacunarity=2).get() #+ 0.3
        elif dusty_type == 'type2':
            noise = generate_fractal_noise_2d((3072, 4096), (1, 1), octaves=8, persistence=0.6, lacunarity=2).get() #+ 0.5
        else:
            noise = generate_fractal_noise_2d((3072, 4096), (1, 1), octaves=8, persistence=0.8, lacunarity=2).get() #+ 0.2

        #生成传输图T
        alpha = random.randint(4,10) * 0.1
        T = 1 - alpha * noise
        
        L = fai * np.max(image)
        T = T.reshape(3072,4096,1)
        H = image * T + L * (1 - T)
        
        H = torch.from_numpy(H.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
        
        H = H / 255
        gt_usm = USMSharp().cuda()(H)
        gt_usm = gt_usm * 255

        ori_h, ori_w = H.size()[2:4]

        # First degradation process
        out = filter2D(gt_usm, self.kernel1)

        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        print(scale)
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        
        
  
        # add noise
        gray_noise_prob = self.opt['gray_noise_prob']

        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = out / 255
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            out = out * 255
        else:
            out = out / 255
            out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            out = out * 255
        

        #print(out)
        # # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        # out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = out / 255  # 将值缩放到 [0, 1] 范围
        out = self.jpeger(out, quality=jpeg_p)
        out = out * 255
        #print(out)
        


        # # Second degradation process
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, self.kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
        
        # # add noise
        gray_noise_prob = self.opt['gray_noise_prob2']        
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            out = out / 255
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            out = out * 255
        else:
                out = out / 255
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
                out = out * 255
        
        #     # JPEG compression + the final sinc filter
        #     # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        #     # as one operation.
        #     # We consider two orders:
        #     #   1. [resize back + sinc filter] + JPEG compression
        #     #   2. JPEG compression + [resize back + sinc filter]
        #     # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = out / 255
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                out = out * 255
                
        else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = out / 255
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                out = out * 255
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                
        return out   # return H 可以输出只经过沙尘合成的图片

if __name__ == '__main__':

    opt_path = '/root/lwk/MarsSR/degradation.yml'
    with open(opt_path, 'r') as f:
        opt = yaml.load(f, Loader=yaml.Loader)

    Data_set = MMSRDataset(opt)
    item = Data_set[0]
    opt_combined = opt.copy()
    opt_combined.update(item)
    # kernel1 = item['kernel1']
    # kernel2 = item['kernel2']
    # sinc_kernel = item['sinc_kernel']

    D = ImageDegradationModule(opt_combined)

    originPath='/root/autodl-tmp/lwk/clean'
    savePath='/root/autodl-tmp/lwk/train/synthesis_lq'
    for image_name in os.listdir(originPath):
        image_path = os.path.join(originPath,image_name)
        real_clean = cv2.imread(image_path)
        lq = D.degrade_image(real_clean)
        lq = lq.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        lq = cv2.normalize(lq, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(savePath,image_name), lq)