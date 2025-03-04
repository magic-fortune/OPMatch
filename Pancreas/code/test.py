import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case
from medpy import metric

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    default="Pancreas_CT",
    help="Dataset to use: LA or Pancreas_CT"
)
parser.add_argument(
    "--root_path",
    type=str,
    default="../data/Pancreas/",
    help="Name of Experiment",
)
parser.add_argument("--model", type=str, default="pct_lab12_re", help="model_name")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument(
    "--epoch_num", type=str, default="best_model", help="checkpoint to use"
)
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
snapshot_path = "../model/" + FLAGS.model + "/"
test_save_path = "../model/prediction/" + FLAGS.model + "_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    stride_xy = 18
    stride_z = 4
    with open(FLAGS.root_path + "/test.list", "r") as f:
        image_list = f.readlines()
    image_list = [
        FLAGS.root_path + item.replace("\n", "") + "/mri_norm2.h5" for item in image_list
    ]
elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    stride_xy = 16
    stride_z = 16
    with open(FLAGS.root_path + "/test.list", "r") as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
else:
    raise ValueError("Invalid dataset name")


def test_calculate_metric(epoch_num="best_model"):
    net = VNet(
        n_channels=1,
        n_classes=num_classes,
        normalization="batchnorm",
        has_dropout=False,
    ).cuda()
    if epoch_num == "best_model":
        save_mode_path = os.path.join(snapshot_path, "best_model_purb.pth")
    else:
        save_mode_path = os.path.join(snapshot_path, "iter_" + str(epoch_num) + ".pth")
    print(f"Using {save_mode_path}")
    net.load_state_dict(torch.load(save_mode_path, weights_only=False))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(
        net,
        image_list,
        num_classes=num_classes,
        patch_size=patch_size,
        stride_xy=stride_xy,
        stride_z=stride_z,
        save_result=True,
        test_save_path=test_save_path,
    )

    return avg_metric


if __name__ == "__main__":
    metric = test_calculate_metric(FLAGS.epoch_num)
