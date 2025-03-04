import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.acdc import ACDCDataset
from model.unet import UNet
from util.classes import CLASSES
from util.utils import AverageMeter, count_params, init_log, DiceLoss
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--ratio', default=1/3, type=float)
parser.add_argument('--scale', default=1, type=float)



def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = UNet(in_chns=1, class_num=cfg['nclass'])    
    model_purb = UNet(in_chns=1, class_num=cfg['nclass'])
    model_purb.load_state_dict(model.state_dict())
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
        
    optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)
    optimizer_purb = SGD(model_purb.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model_purb.cuda()   

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg['nclass'])
    
    trainset_u = ACDCDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = ACDCDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    trainsampler_u_mix = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u_mix = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u_mix)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    previous_best_purb = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    def add_gaussian_noise(optimizer, noise_std=1e-4):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * noise_std
                    param.grad.data += noise
                    
    def add_noise(optimizer, noise_type="gaussian", stddev=0.1, noise_strength=0.01):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if noise_type == "gaussian" or noise_type == "g": 
                    noise = torch.randn_like(param) * stddev  
                elif noise_type == "uniform"  or noise_type == "u":
                    noise = (torch.rand_like(param) - 0.5) * 2 * stddev  
                elif noise_type == "laplace"  or noise_type == "l":
                    noise = torch.randn_like(param) * torch.sign(torch.randn_like(param)) * stddev  
                else:
                    raise ValueError("Unsupported noise type")
                param.data += noise
        
                    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))
            
        add_noise(optimizer_purb, noise_type='g', stddev=args.scale)
        
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        trainloader_u_mix.sampler.set_epoch(epoch + cfg['epochs'])
        
        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
             
            with torch.no_grad():
                model.eval()
                model_purb.eval()
                
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                
                pred_u_w_mix_purb = model_purb(img_u_w_mix).detach()
                conf_u_w_mix_purb = pred_u_w_mix_purb.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix_purb = pred_u_w_mix_purb.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()
            model_purb.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            
            preds_purb, preds_fp_purb = model_purb(torch.cat((img_x, img_u_w)), True)
            pred_x_purb, pred_u_w_purb = preds_purb.split([num_lb, num_ulb])
            pred_u_w_fp_purb = preds_fp_purb[num_lb:]
            

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)
            pred_u_s1_purb, pred_u_s2_purb = model_purb(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
            
            pred_u_w_purb_de = pred_u_w_purb.detach()
            conf_u_w_purb_de = pred_u_w_purb_de.softmax(dim=1).max(dim=1)[0]
            mask_u_w_purb_de = pred_u_w_purb_de.argmax(dim=1)


            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1_purb, conf_u_w_cutmixed1_purb = mask_u_w_purb_de.clone(), conf_u_w_purb_de.clone()
            mask_u_w_cutmixed2_purb, conf_u_w_cutmixed2_purb = mask_u_w_purb_de.clone(), conf_u_w_purb_de.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

            mask_u_w_cutmixed1_purb[cutmix_box1 == 1] = mask_u_w_mix_purb[cutmix_box1 == 1]
            conf_u_w_cutmixed1_purb[cutmix_box1 == 1] = conf_u_w_mix_purb[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2_purb[cutmix_box2 == 1] = mask_u_w_mix_purb[cutmix_box2 == 1]
            conf_u_w_cutmixed2_purb[cutmix_box2 == 1] = conf_u_w_mix_purb[cutmix_box2 == 1]

            loss_x = (criterion_ce(pred_x, mask_x) + criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())) / 2.0

            loss_u_s1 = criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']).float())
            
            loss_u_s2 = criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed2 < cfg['conf_thresh']).float())
            
            loss_u_w_fp = criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1).float(),
                                         ignore=(conf_u_w < cfg['conf_thresh']).float())
            
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            loss_add = (criterion_dice(pred_u_s1_purb.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']).float()) + \
                        criterion_dice(pred_u_s2_purb.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed2 < cfg['conf_thresh']).float()) ) / 2.0
            

            loss_x_purb = (criterion_ce(pred_x_purb, mask_x) + criterion_dice(pred_x_purb.softmax(dim=1), mask_x.unsqueeze(1).float())) / 2.0

            loss_u_s1_purb = criterion_dice(pred_u_s1_purb.softmax(dim=1), mask_u_w_cutmixed1_purb.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed1_purb < cfg['conf_thresh']).float())
            
            loss_u_s2_purb = criterion_dice(pred_u_s2_purb.softmax(dim=1), mask_u_w_cutmixed2_purb.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed2_purb < cfg['conf_thresh']).float())
            
            loss_u_w_fp_purb = criterion_dice(pred_u_w_fp_purb.softmax(dim=1), mask_u_w_purb_de.unsqueeze(1).float(),
                                         ignore=(conf_u_w_purb_de < cfg['conf_thresh']).float())
            
            loss_purb = (loss_x_purb + loss_u_s1_purb * 0.25 + loss_u_s2_purb * 0.25 + loss_u_w_fp_purb * 0.5  + loss_add * args.ratio) / 2.0
            
            optimizer_purb.zero_grad()
            loss_purb.backward()
            optimizer_purb.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = (conf_u_w >= cfg['conf_thresh']).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())
            
            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            # optimizer_purb.param_groups[0]["lr"] = lr
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, 
                                            total_loss_w_fp.avg, total_mask_ratio.avg))

        model.eval()
        model_purb.eval()
        dice_class = [0] * 3
        dice_class_purb = [0] * 3
        
        with torch.no_grad():
            for img, mask in valloader:
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

                img = img.permute(1, 0, 2, 3)
                
                pred = model(img)
                pred_purb = model_purb(img)
                
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)
                
                pred_purb = F.interpolate(pred_purb, (h, w), mode='bilinear', align_corners=False)
                pred_purb = pred_purb.argmax(dim=1).unsqueeze(0)

                for cls in range(1, cfg['nclass']):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls-1] += 2.0 * inter / union
                    
                    inter_purb = ((pred_purb == cls) * (mask == cls)).sum().item()
                    union_purb = (pred_purb == cls).sum().item() + (mask == cls).sum().item()
                    dice_class_purb[cls-1] += 2.0 * inter_purb / union_purb

        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)
        
        dice_class_purb = [dice * 100.0 / len(valloader) for dice in dice_class_purb]
        mean_dice_purb = sum(dice_class_purb) / len(dice_class_purb)
        
        if rank == 0:
            for (cls_idx, dice) in enumerate(dice_class):
                logger.info('***** Evaluation model ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
            logger.info('***** Evaluation nodel ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))
            
            for (cls_idx, dice) in enumerate(dice_class_purb):
                logger.info('***** Evaluation model_purb ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
            logger.info('***** Evaluation model_purb ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice_purb))
            
            
            writer.add_scalar('eval/MeanDice', mean_dice, epoch)
            writer.add_scalar('eval/MeanDice_purb', mean_dice_purb, epoch)
            
            for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_dice' % (CLASSES[cfg['dataset']][i]), dice, epoch)
            
            for i, dice in enumerate(dice_class_purb):
                writer.add_scalar('eval/%s_dice_purb' % (CLASSES[cfg['dataset']][i]), dice, epoch)
            
        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        
        is_best_purb = mean_dice_purb > previous_best_purb
        previous_best_purb = max(mean_dice_purb, previous_best_purb)
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            checkpoint_purb = {
                'model': model_purb.state_dict(),
                'optimizer': optimizer_purb.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best_purb,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            torch.save(checkpoint_purb, os.path.join(args.save_path, 'latest_purb.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            if is_best_purb:
                torch.save(checkpoint_purb, os.path.join(args.save_path, 'best_purb.pth'))


if __name__ == '__main__':
    main()
