import os
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from utils.losses import dice_loss
from dataloaders.la_heart import LAHeart, RandomCrop, RandomRotFlip, ToTensor
from dataloaders.pancreas_ct import Pancreas
from test_util import test_all_case
from utils import ramps
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    default="LA",
    help="Dataset to use: LA or Pancreas_CT",
)
parser.add_argument(
    "--root_path",
    type=str,
    default="../data/2018LA_Seg_Training Set/",
    help="Name of Experiment",
)
parser.add_argument("--exp", type=str, default="distill_match", help="model_name")
parser.add_argument(
    "--max_iterations", type=int, default=9000, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=2, help="batch_size per gpu")
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="maximum epoch number to train"
)
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--label_num", type=int, default=16, help="label num")
parser.add_argument("--eta", type=float, default=0.3, help="weight to balance loss")
parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer")
parser.add_argument("--conf_thresh", type=float, default=0.85, help="conf_thresh")
parser.add_argument("--temperature", type=float, default=1, help="temperature")
parser.add_argument("--size", type=float, default=1e-3, help="purb size")

parser.add_argument(
    "--loss-mix-eval",
    default="(loss_mix_img_u + loss_mix_img_x + loss_mix_feat_u + loss_mix_feat_x) / 4.0",
    type=str,
)
parser.add_argument(
    "--loss-eval",
    default="(loss_x + loss_mix * (1-args.eta) + loss_s * args.eta) / 2.0",
    type=str,
)

parser.add_argument("--feature_alpha", default=0.55, type=float)
parser.add_argument("--image_alpha", default=0.80, type=float)

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(","))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
else:
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

patch_size = (112, 112, 80)
num_classes = 2

LABELED_ID_NUM = args.label_num  # 8 or 16
conf_thresh = args.conf_thresh
eta = args.eta
pervious_bset_dice = 0.0
pervious_bset_dice_purb = 0.0

if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = '../data/2018LA_Seg_Training Set/'
    args.max_samples = 80
    DATASET_CLASS = LAHeart
    TSFM = transforms.Compose(
        [
            RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ]
    )
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = '../data/Pancreas/'
    args.max_samples = 62
    args.max_samples = 62
    DATASET_CLASS = Pancreas
    TSFM = transforms.Compose(
        [
            RandomCrop(patch_size),
            ToTensor(),
        ]
    )

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")

    shutil.copytree(
        ".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"])
    )

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = VNet(
        n_channels=1, n_classes=num_classes, normalization="batchnorm", has_dropout=True
    )
    net_purb = VNet(
        n_channels=1, n_classes=num_classes, normalization="batchnorm", has_dropout=True
    )
    net = net.cuda()
    net_purb = net_purb.cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        
    trainset_u = DATASET_CLASS(
        base_dir=train_data_path,
        mode="train_u",
        num=args.max_samples - LABELED_ID_NUM,
        transform=TSFM,
        id_path=f"{args.root_path}/train_{LABELED_ID_NUM}_unlabel.list",
    )
    trainsampler_u = torch.utils.data.sampler.RandomSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_u,
        worker_init_fn=worker_init_fn
    )
    trainsampler_u_mix = torch.utils.data.sampler.RandomSampler(trainset_u, replacement=True)
    trainloader_u_mix = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_u_mix,
        worker_init_fn=worker_init_fn
    )

    trainset_l = DATASET_CLASS(
        base_dir=train_data_path,
        mode="train_l",
        num=args.max_samples - LABELED_ID_NUM,
        transform=TSFM,
        id_path=f"{args.root_path}/train_{LABELED_ID_NUM}_label.list",
    )
    trainsampler_l = torch.utils.data.sampler.RandomSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_l,
        worker_init_fn=worker_init_fn
    )

    net.train()
    net_purb.train()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
        )
        optimizer_purb = optim.SGD(
            net_purb.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
        )
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
        optimizer_purb = optim.Adam(net_purb.parameters(), lr=base_lr, weight_decay=0.0001)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)
        optimizer_purb = optim.AdamW(net_purb.parameters(), lr=base_lr, weight_decay=0.0001)
    else:
        raise NotImplementedError
    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} itertations per epoch".format(len(trainloader_l)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_l) + 1
    print(f"All Epochs: {max_epoch}")
    
    lr_ = base_lr
    net.train()
    net_purb.train()
    
            
    def add_noise(optimizer, noise_type="gaussian", stddev=0.1, noise_strength=0.01):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if noise_type == "gaussian" or noise_type == "g": 
                    noise = torch.randn_like(param) * stddev  
                elif noise_type == "uniform"  or noise_type == "u":
                    noise = (torch.rand_like(param) - 0.5) * 2 * stddev  
                elif noise_type == "laplace"  or noise_type == "l":
                    noise = torch.randn_like(param) * torch.sign(torch.randn_like(param)) * stddev  
                elif noise_type == "bernoulli"  or noise_type == "b":
                    noise = (torch.rand_like(param) < 0.5).float() * noise_strength  
                else:
                    raise ValueError("Unsupported noise type")
                param.data += noise



    for epoch_num in tqdm(range(max_epoch), ncols=70):
        
        add_noise(optimizer_purb, noise_type="g", stddev=1e-3, noise_strength=0.01)
        net.train()
        net_purb.train()

        def loss_x_fn(pred, mask):
            return (
                F.cross_entropy(pred, mask)
                + dice_loss(pred.softmax(dim=1)[:, 1, :, :, :], mask == 1)
            ) / 2.0

        def loss_u_fn(pred, mask, conf):
            return dice_loss(
                pred.softmax(dim=1)[:, 1, :, :, :],
                mask == 1,
                ignore=(conf < conf_thresh).float(),
            )

        for i_batch, ((img_x, mask_x), (img_u_w, img_u_s, img_u_s2, _, _)) in enumerate(
            zip(trainloader_l, trainloader_u)
        ):
           
            img_x, mask_x = img_x.cuda(non_blocking=True), mask_x.cuda(
                non_blocking=True
            )
            img_u_w = img_u_w.cuda(non_blocking=True)
            img_u_s = img_u_s.cuda(non_blocking=True)
            img_u_s2 = img_u_s2.cuda(non_blocking=True)

            iters = epoch_num * len(trainloader_u) + i_batch
            
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            preds, preds_fp = net(torch.cat((img_x, img_u_w)), True)
            preds_purb, preds_fp_purb = net_purb(torch.cat((img_x, img_u_w)), True)
            
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_x_purb, pred_u_w_purb = preds_purb.split([num_lb, num_ulb])
            
            pred_u_w_fp = preds_fp[num_lb:]
            pred_u_w_fp_purb = preds_fp_purb[num_lb:]
            
            pred_u_s1, pred_u_s2 = net(torch.cat((img_u_s, img_u_s2))).chunk(2)
            pred_u_s1_purb, pred_u_s2_purb = net_purb(torch.cat((img_u_s, img_u_s2))).chunk(2)
            
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
            
            pred_u_w_purb_de = pred_u_w_purb.detach()
            conf_u_w_purb_de = pred_u_w_purb_de.softmax(dim=1).max(dim=1)[0]
            mask_u_w_purb_de = pred_u_w_purb_de.argmax(dim=1)

            loss_x = loss_x_fn(pred_x, mask_x)
            loss_u_s1 = loss_u_fn(pred_u_s1, mask_u_w, conf_u_w)
            loss_u_s2 = loss_u_fn(pred_u_s2, mask_u_w, conf_u_w)
            loss_s = loss_u_fn(pred_u_w_fp, mask_u_w, conf_u_w)

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_s * 0.5) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred_u_s_de = pred_u_s1.detach()
            conf_u_s_de = pred_u_s_de.softmax(dim=1).max(dim=1)[0]
            mask_u_s_de = pred_u_s_de.argmax(dim=1)
                   
            pred_u_s2_de = pred_u_s2.detach()
            conf_u_s2_de = pred_u_s2_de.softmax(dim=1).max(dim=1)[0]
            mask_u_s2_de = pred_u_s2_de.argmax(dim=1)
            
            pred_u_w_fp_de = pred_u_w_fp.detach()
            conf_u_w_fp_de = pred_u_w_fp_de.softmax(dim=1).max(dim=1)[0]
            mask_u_w_fp_de = pred_u_w_fp_de.argmax(dim=1)
            
            loss_x_purb = loss_x_fn(pred_x_purb, mask_x)
            
            loss_u_s1_purb = loss_u_fn(pred_u_s1_purb, mask_u_w_purb_de, conf_u_w_purb_de)
            loss_u_s2_purb = loss_u_fn(pred_u_s2_purb, mask_u_w_purb_de, conf_u_w_purb_de)
            loss_s_purb = loss_u_fn(pred_u_w_fp_purb, mask_u_w_purb_de, conf_u_w_purb_de)
            
            loss_add_purb = (loss_u_fn(pred_u_s1_purb, mask_u_w, conf_u_w) + loss_u_fn(pred_u_s2_purb, mask_u_w, conf_u_w)) / 2.0
            
            loss_purb = (loss_x_purb + loss_u_s1_purb * 0.25 + loss_u_s2_purb * 0.25 + loss_s_purb * 0.5 + loss_add_purb * 0.5) / 2.0
            
            optimizer_purb.zero_grad()
            loss_purb.backward()
            optimizer_purb.step()
            
            mask_ratio = (conf_u_w >= conf_thresh).sum() / conf_u_w.numel()

            conf_thresh = (
                args.conf_thresh
                + (1 - args.conf_thresh)
                * ramps.sigmoid_rampup(iter_num, max_iterations)
            ) * np.log(2)

            iter_num = iter_num + 1
            writer.add_scalar("train/loss_all", loss.item(), iters)
            writer.add_scalar("train/loss_x", loss_x.item(), iters)
            writer.add_scalar("train/loss_s", loss_s.item(), iters)
            writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            lr_ = base_lr * (1 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_
                
            # if iter_num % 50 == 0:
            #     image = (
            #         img_x[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     )
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image("train/Image", grid_image, iter_num)

            #     outputs_soft = F.softmax(pred_x, 1)
            #     image = (
            #         outputs_soft[0, 1:2, :, :, 20:61:10]
            #         .permute(3, 0, 1, 2)
            #         .repeat(1, 3, 1, 1)
            #     )
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image("train/Predicted_label", grid_image, iter_num)

            #     image = (
            #         mask_x[0, :, :, 20:61:10]
            #         .unsqueeze(0)
            #         .permute(3, 0, 1, 2)
            #         .repeat(1, 3, 1, 1)
            #     )
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image("train/Groundtruth_label", grid_image, iter_num)

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "iter_" + str(iter_num) + ".pth"
                )
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                print("finish training, iter_num > max_iterations")
                break
        if iter_num > max_iterations:
            print("finish training")
            break

        if  epoch_num > max_epoch // 2 or epoch_num == 10:
            # evals
            net.eval()
            net_purb.eval()
            with torch.no_grad():
                with open(args.root_path + "./test.list", "r") as f:
                    image_list = f.readlines()
                if args.dataset_name == "LA":
                    image_list = [
                        args.root_path + item.replace("\n", "") + "/mri_norm2.h5"
                        for item in image_list
                    ]

                    dice, jc, hd, asd = test_all_case(
                        net,
                        image_list,
                        num_classes=num_classes,
                        patch_size=patch_size,
                        stride_xy=18,
                        stride_z=4,
                        save_result=False,
                        test_save_path=None,
                    )
                elif args.dataset_name == "Pancreas_CT":
                    image_list = [args.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]

                    # dice, jc, hd, asd = test_all_case(
                    #     net,
                    #     image_list,
                    #     num_classes=num_classes,
                    #     patch_size=patch_size,
                    #     stride_xy=16,
                    #     stride_z=16,
                    #     save_result=False,
                    #     test_save_path=None,
                    # )
                    
                    dice_purb, jc_purb, hd_purb, asd_purb = test_all_case(
                        net_purb,
                        image_list,
                        num_classes=num_classes,
                        patch_size=patch_size,
                        stride_xy=16,
                        stride_z=16,
                        save_result=False,
                        test_save_path=None,
                    )

                # if dice > pervious_bset_dice:
                #     pervious_bset_dice = dice
                #     save_mode_path = os.path.join(snapshot_path, "best_model.pth")
                #     torch.save(net.state_dict(), save_mode_path)
                #     logging.info("save model to {}".format(save_mode_path))

                if dice_purb > pervious_bset_dice_purb:
                    pervious_bset_dice_purb = dice_purb
                    save_mode_path_purb = os.path.join(snapshot_path, "best_model_purb.pth")
                    torch.save(net_purb.state_dict(), save_mode_path_purb)
                    logging.info("save model to {}".format(save_mode_path_purb))

    save_mode_path = os.path.join(
        snapshot_path, "iter_" + str(max_iterations + 1) + ".pth"
    )
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
