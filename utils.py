import random
import numpy as np
import torch
import os
import shutil
from ptflops import get_model_complexity_info
import time
from skimage import color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# 设置随机种子 目的是让结果是可以复现
def set_random_seed(seed, deterministic=False):
    # 带有random的初始化
    random.seed(seed)
    # 带有numpy的初始化
    np.random.seed(seed)
    # 带有torch的初始化，包括了模型与数据集的初始化
    torch.manual_seed(seed)
    # 在所有 CUDA 设备上设置随机数生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_dir(results_dir='./results', experiment='experiments', delete=True):
    # 创建模型、日志、训练、验证文件夹
    models_dir = os.path.join(results_dir, experiment, 'models')
    log_dir = os.path.join(results_dir, experiment, 'log')
    train_images_dir = os.path.join(results_dir, experiment, 'images', 'train')
    val_images_dir = os.path.join(results_dir, experiment, 'images', 'val')

    # 如果delete为True，则删除文件夹下的文件
    clean_dir(models_dir, delete=delete)
    clean_dir(log_dir, delete=delete)
    clean_dir(train_images_dir, delete=delete)
    clean_dir(val_images_dir, delete=delete)
    
    return models_dir, log_dir, train_images_dir, val_images_dir

def clean_dir(path, delete, contain=False):
    # 如果传递过来的path不存在，则创建文件夹 就不需要进行删除操作了
    if not os.path.exists(path):
        os.makedirs(path)
    # 如果已经存在了 那么就要进行清理操作
    elif delete:
        delete_under(path, contain=contain)

def delete_under(path, contain=False):
    # 如果contain为True，则删除文件夹下的所有文件，包括这个path本身的文件夹
    if contain:
        shutil.rmtree(path)
    # 如果contain为False，则删除文件夹下的所有文件，不包括这个path本身的文件夹
    else:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # print("删除了")
                # print(type(file_path))
                shutil.rmtree(file_path)


def print_flops(model, input_size=(3, 256, 256)):
    # 计算并打印FLOPs
    flops, _ = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False)
    print(f"FLOPs: {flops}")

def print_para_num(model):
    # 打印模型的所有参数
    total_params = sum(p.numel() for p in model.parameters())
    # 打印所有可以训练的参数
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters: %d' % total_params)
    print('trainable parameters: %d' % total_trainable_params)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    def average(self, auto_reset=False):
        # 如果count为0，返回0，否则返回sum/count
        avg = self.sum / self.count

        # 在每个batch结束之后更新average
        if auto_reset:
            self.reset() 

        return avg

class Timer(object):
    def __init__(self, start=True):
        if start:
            self.start()
    
    def start(self):
        self.time_begin = time.time()

    def timeit(self, auto_reset=True):
        times = time.time() - self.time_begin
        # 每次打印出之后再更新
        if auto_reset:
            self.start()
        return times

def tensor2img(tensor_image):
    
    tensor_image = tensor_image*255
    # BCHW->BHWC
    tensor_image = tensor_image.permute([0, 2, 3, 1])
    if tensor_image.device != 'cpu':
        tensor_image = tensor_image.cpu()
    # 把tensor转换为numpy
    numpy_image = np.uint8(tensor_image.numpy())
    return numpy_image

def get_metrics(tensor_image1, tensor_image2, psnr_only=True, reduction=False):
    # 第一张是predict 第二张是gts
    # 计算评价指标
    if len(tensor_image1.shape) != 4 or len(tensor_image2.shape) != 4:
        raise Exception('a batch tensor image pair should be given!')
    
    numpy_imgs = tensor2img(tensor_image1)
    numpy_gts = tensor2img(tensor_image2)

    psnr_value, ssim_value = 0., 0.
    batch_size = numpy_imgs.shape[0]

    for i in range(batch_size):
    # 转换为YCbCr色彩空间，然后只保留Y通道
        y_img = color.rgb2ycbcr(numpy_imgs[i])[:, :, 0]
        y_gt = color.rgb2ycbcr(numpy_gts[i])[:, :, 0]
        psnr_value += peak_signal_noise_ratio(y_img, y_gt, data_range=255)
        ssim_value += structural_similarity(y_img, y_gt, data_range=255, multichannel=False)

    if reduction:
        psnr_value = psnr_value/batch_size
        ssim_value = ssim_value/batch_size
    
    if not psnr_only:  
        return psnr_value, ssim_value
    else:
        return psnr_value

def check_padding(x):
    h, w = x.size(2), x.size(3)
    meta_size = 2
    if (h % 2) == 0:
        h_pad = 0
    else:
        h_pad = meta_size - (h % 2)
    if (w % 2) == 0:
        w_pad = 0
    else:
        w_pad = meta_size - (w % 2)
    pad = torch.nn.ZeroPad2d(padding=(0, w_pad, 0, h_pad))
    x = pad(x)
    return x