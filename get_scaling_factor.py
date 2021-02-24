import torch
from imageio import imread
import numpy as np
from utils import get_factor
import models
import argparse

parser = argparse.ArgumentParser(description='Script for computing the scaling factor',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dispnet', dest='dispnet', required=True, type=str, choices=['DispNet', 'DispResNet'],
                    help='depth network architecture.')
parser.add_argument("--pretrained-dispnet", required=True,
                    type=str, help="pretrained DispNet path")
parser.add_argument("--frame-path", type=str, help="Path to the frame")
parser.add_argument("--intrinsics-path", type=str, help="Path to intrinsics file")

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_tensor_image(filename):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    img = np.transpose(img, (2, 0, 1))
    tensor_img = (
        (torch.from_numpy(img).unsqueeze(0)/255 - 0.5)/0.5).to(device)
    return tensor_img


def main():
    args = parser.parse_args()
    img = load_tensor_image(args.frame_path)

    disp_net = getattr(models, args.dispnet)().to(device)
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    camera_matrix = np.loadtxt(args.intrinsics_path)

    print(camera_matrix)

    depth = 1 / disp_net(img).cpu().detach().double()

    print(depth.shape)
    
    scaling_factor = get_factor(depth, camera_matrix)

    print(scaling_factor)


if __name__ == "__main__":
    main()
