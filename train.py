import time
import torch
import random

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn as nn
import os
from torch.utils.data import DataLoader
import wandb
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

from utils import *
from options import TrainOptions
from model import Restormer
from losses import LossCont, LossFreqReco, Losslpips
from dataset import PairImagesDataset, ValImgDataset

# 如果有多块GPU 设置使用第2和3块的GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')
opt = TrainOptions().parse()
set_random_seed(opt.seed)

# delete=(not opt.resume)如果命令行中有--resume说明不会删掉之前的文件
models_dir, log_dir, train_images_dir, val_images_dir = prepare_dir(opt.results_dir, opt.experiment, delete=(not opt.resume))
writer = SummaryWriter(log_dir=log_dir)

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
print('training data loading...')
train_dataset = PairImagesDataset(datasource=opt.data_source + '/train', crop=opt.crop)
print('successfully loading training pairs. =====> qty:{} bs:{}'.format(len(train_dataset),opt.train_bs))
print('validating data loading...')
val_dataset = ValImgDataset(datasource=opt.data_source + '/val')
val_dataloader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(val_dataset),opt.val_bs))
print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = Restormer().cuda()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 这将模型复制到所有可用的GPU
    model = nn.DataParallel(model)

print_para_num(model)
print_flops(model, input_size=(3, 256, 256))

if opt.pretrained is not None:
    model.load_state_dict(torch.load(opt.pretrained))
    print('successfully loading pretrained model.')

print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
criterion_cont = LossCont()
criterion_fft = LossFreqReco()
criterion_lpips = Losslpips()

optimizer = AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

print('---------------------------------------- step 5/5 : training... ----------------------------------------------------')
# best_lpips_epoch = []
def save_checkpoint(model, best_lpips_score, epoch):

    model_filename = f'best-lpips-{epoch}-{best_lpips_score}-ckpt.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_lpips_score': best_lpips_score,
    }, os.path.join(models_dir, model_filename))
    print(f"Best model saved with LPIPS: {best_lpips_score} at epoch {epoch}")

# 全局变量定义
global train_dataloader
train_dataloader = None

def prepare_dataloader(epoch, batch_size, crop_size):
    global train_dataloader
    train_dataset = PairImagesDataset(datasource=opt.data_source + '/train', crop=crop_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    print('DataLoader updated: bs:{} crop:{}'.format(batch_size, crop_size))

def main():
    # 初始化wandb
    wandb.init(project="Restormer", name='restormer-resize')
    wandb.config.update(opt)

    best_lpips_score = float('inf')  # Initialize with a high value
    # print("Scheduler milestones:", scheduler.milestones)  # 打印学习率调度器的里程碑

    start_epoch = 1
    if opt.resume:
        state = torch.load(models_dir + '/latest.pth')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        print('Resume from epoch %d' % (start_epoch))
        #print("Scheduler milestones:", scheduler.milestones)  # 打印学习率调度器的里程碑
    
    for epoch in range(start_epoch, opt.n_epochs + 1):
        train(epoch)
        # 在第100个epoch之后写开始准备保存val的权重
        if epoch >= 100:
            if epoch % opt.val_gap == 0:
                current_lpips_score = val(epoch)
                print('current_lpips_score:', current_lpips_score)
                print('best_lpips_score:', best_lpips_score)
                is_best = current_lpips_score < best_lpips_score
                if is_best:
                    best_lpips_score = current_lpips_score
                    save_checkpoint(model, best_lpips_score, epoch)
                # best_lpips_epoch.append((epoch, best_lpips_score))
    
    writer.close()
    wandb.finish()


def train(epoch):
    torch.cuda.empty_cache()
    model.train()

    lpips_meter = AverageMeter()
    psnr_meter = AverageMeter()    

    iter_cont_meter = AverageMeter()
    iter_fft_meter = AverageMeter()
    iter_timer = Timer()

    global train_dataloader
    if epoch >= 1 and epoch < 60:
        prepare_dataloader(epoch, 16, 128)
    elif epoch >= 60 and epoch < 120:
        prepare_dataloader(epoch, 12, 160)
    elif epoch >= 120 and epoch < 180:
        prepare_dataloader(epoch, 8, 192)
    elif epoch >= 180 and epoch < 240:
        prepare_dataloader(epoch, 4, 256)
    elif epoch >= 240 and epoch <= opt.n_epochs:
        prepare_dataloader(epoch, 2, 384)
    
    max_iter = len(train_dataloader)

    for i, (gts, flare) in enumerate(train_dataloader):
        flare = flare.cuda()
        gts = gts.cuda()
        cur_batch = flare.shape[0]
        h, w = flare.size(2), flare.size(3)
        flare = check_padding(flare)

        optimizer.zero_grad()
        preds = model(flare)

        preds = torch.clamp(preds, 0, 1)
        preds = preds[:, :, :h, :w]

        loss_cont = criterion_cont(preds, gts)
        loss_fft = criterion_fft(preds, gts)
        loss_lpips = criterion_lpips(preds, gts)

        loss = loss_cont + opt.lambda_fft * loss_fft + loss_lpips

        loss.backward()
        optimizer.step()

        psnr_meter.update(get_metrics(torch.clamp(preds.detach(), 0, 1), gts), cur_batch)

        lpips_meter.update(loss_lpips.item()*cur_batch, cur_batch)
        iter_cont_meter.update(loss_cont.item()*cur_batch, cur_batch)
        iter_fft_meter.update(loss_fft.item()*cur_batch, cur_batch)

        if i == 0:
            save_image(torch.cat((flare,preds.detach(),gts),0), train_images_dir + '/epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.train_bs, normalize=True, scale_each=True)

        if i % opt.print_gap == 0:
            print('Training: Epoch[{:0>4}/{:0>4}] Iteration[{:0>4}/{:0>4}] Loss_lipips: {:.4f} Loss_cont: {:.4f} Loss_fft: {:.4f} PSNR: {:.4f} LPIPS: {:.4f} Time: {:.4f}'.format(epoch, opt.n_epochs, i + 1, max_iter, lpips_meter.average(),iter_cont_meter.average(), iter_fft_meter.average(), psnr_meter.average(), lpips_meter.average(), iter_timer.timeit()))
            wandb.log({'LPIPS': lpips_meter.average(auto_reset=True), 'PSNR': psnr_meter.average(auto_reset=True),'Loss_cont': iter_cont_meter.average(auto_reset=True),
           'Loss_fft': iter_fft_meter.average(auto_reset=True)}, step=i+1 + (epoch - 1) * max_iter)
            wandb.log({'lr': scheduler.get_last_lr()[0]}, step=i+1 + (epoch - 1) * max_iter)
        
        if (epoch == 1 and i == 0) or (epoch == 60 and i == 0) or (epoch == 120 and i == 0) or (epoch == 180 and i == 0) or (epoch == 240 and i == 0) or (epoch == 300 and i == 0):
            wandb.log({"img-train": [wandb.Image(make_grid(flare, normalize=True),caption='flare'), 
                                     wandb.Image(make_grid(preds, normalize=True),caption='preds'), 
                                     wandb.Image(make_grid(gts, normalize=True),caption='gts')]})
    
    
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}, models_dir + '/latest.pth')
    scheduler.step()

def val(epoch):
    torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        model_single = model.module
    else:
        model_single = model

    model_single.eval()
    print(''); print('Validating...', end=' ')

    timer = Timer()
    lpips_meter = AverageMeter()
    psnr_meter = AverageMeter()

    for i, (gts, flare) in enumerate(val_dataloader):
        gts, flare = gts.cuda(), flare.cuda()
        cur_batch = flare.shape[0]
        h, w = flare.size(2), flare.size(3)
        flare = check_padding(flare)

        with torch.no_grad():
            preds = model_single(flare)
            preds = torch.clamp(preds, 0, 1)
            preds = preds[:, :, :h, :w]
            flare = flare[:, :, :h, :w]

            lpips_value = criterion_lpips(preds, gts).mean()
            lpips_meter.update(lpips_value.item() * cur_batch, cur_batch)

            psnr_value = get_metrics(torch.clamp(preds, 0, 1), gts)
            psnr_meter.update(get_metrics(preds, gts), cur_batch)
        
        if i == 0:
            save_image(torch.cat((flare, preds.detach(), gts), 0), 
                       os.path.join(val_images_dir, f'epoch_{epoch:04d}_iter_{i+1:04d}.png'), 
                       nrow=opt.val_bs, normalize=True, scale_each=True)
            wandb.log({"img-validation": [wandb.Image(make_grid(flare, normalize=True),caption='flare'), 
                                     wandb.Image(make_grid(preds, normalize=True),caption='preds'), 
                                     wandb.Image(make_grid(gts, normalize=True),caption='gts')]})


        if i % opt.print_gap == 0:
            print('Validation: Epoch[{:0>4}] Iteration[{:0>4}/{:0>4}] PSNR: {:.4f} LPIPS: {:.4f}'.format(epoch, i + 1, len(val_dataloader), psnr_value, lpips_meter.average()))
    # wandb.log({'Val-LPIPS': lpips_meter.average()},step=global_step)
    # wandb.log({'Val-PSNR': psnr_meter.average()},step=global_step)
    avg_lpips = lpips_meter.average()
    writer.add_scalar('val-lpips', avg_lpips, epoch)
    writer.add_scalar('val-psnr', psnr_meter.average(), epoch)

    print('Epoch[{:0>4}/{:0>4}] Time: {:.4f}'.format(epoch, opt.n_epochs, timer.timeit())); print('')
    
    return lpips_meter.average()

if __name__ == '__main__':
    main()       