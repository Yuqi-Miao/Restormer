import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class LossCont(nn.Module):
    def __init__(self):
        # 因后续需要用到nn.L1Loss()，所以需要先调用父类的构造函数
        super(LossCont, self).__init__()
        # L1 损失计算两个输入元素间的绝对差值的平均值。
        self.criterion = nn.L1Loss()
        
    def forward(self, imgs, gts):
        # 相当于在重写__call__方法中的forward函数
        return self.criterion(imgs, gts)

class LossFreqReco(nn.Module):
    def __init__(self):
        super(LossFreqReco, self).__init__()
        # 本质上还是用的L1损失 但是在计算之前需要先将输入张量进行傅里叶变换
        # 是在频域内计算损失
        self.criterion = nn.L1Loss()
        
    def forward(self, imgs, gts):
        # 将输入张量的实部和虚部分别拆分出来
        imgs = torch.fft.rfftn(imgs, dim=(2,3))
        _real = imgs.real
        _imag = imgs.imag
        imgs = torch.cat([_real, _imag], dim=1)
        gts = torch.fft.rfftn(gts, dim=(2,3))
        _real = gts.real
        _imag = gts.imag
        gts = torch.cat([_real, _imag], dim=1)
        return self.criterion(imgs, gts)

class Losslpips(nn.Module):
    def __init__(self):
        super(Losslpips, self).__init__()
        self.lpips_loss = lpips.LPIPS(net='alex').cuda()
        
    def forward(self, imgs, gts):
        return self.lpips_loss(imgs, gts).mean()