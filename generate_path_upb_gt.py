import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from numpy.linalg import inv
from skimage.transform import resize as imresize
import torch
import models
from imageio import imread, imsave
import argparse
from path import Path
from data import kitti_odom_loader
from PIL import Image
import time
import os
from inverse_warp import pose_mat2vec, pose_vec2mat
from tqdm import tqdm
import seaborn as sns
from inverse_warp import inverse_warp2

parser = argparse.ArgumentParser(description='Script for visualizing the path',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", type=str, help="pretrained DispNet path",
                    default='/home/andrei/workspace/nemodrive/SC-SfMLearner-Release/checkpoints/papers/cs+k_depth.tar')
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
parser.add_argument("--dataset-dir", type=str, help="Dataset directory",
                    default='/mnt/storage/workspace/andreim/nemodrive/upb_data/dataset/validation_frames/')
parser.add_argument("--scale-translation", default=False,
                    nargs='*', type=bool, help="specifies if the translation is scaled to gt")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'],
                    nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--segmentation-path",
                    default="/mnt/storage/workspace/andreim/nemodrive/upb_self_supervised_labels/",
                    nargs='*', type=str, help="path to segmentation dataset")

CAR_L = 2.634
WHEEL_STEER_RATIO = 18.215151515151515
MAX_R = 9999999999.9999999

def get_radius(wheel_angle, car_l=CAR_L):
    if wheel_angle != 0.0:
        r = car_l / np.tan(np.deg2rad(wheel_angle, dtype=np.float64))
    else:
        r = MAX_R

    return r


def rotate_point(cx, cy, angle, px, py):
    s = np.sin(angle);
    c = np.cos(angle);

    # translate point back to origin:
    px -= cx;
    py -= cy;

    # rotate point
    xnew = px * c - py * s;
    ynew = px * s + py * c;

    # translate point back:
    px = xnew + cx;
    py = ynew + cy;
    return px, py;


@torch.no_grad()
def main():
    args = parser.parse_args()

    limit_1 = 8
    limit_2 = 28

    sequences = os.listdir(args.dataset_dir)

    for seq in sequences:
        if '.txt' not in seq:
            print(seq)
            args.sequence = seq

            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            image_dir = Path(args.dataset_dir + args.sequence)

            test_files = sum([image_dir.files('*.{}'.format(ext))
                              for ext in args.img_exts], [])
            test_files.sort()
            test_files = test_files
            print('{} files to test'.format(len(test_files)))

            camera_matrix = np.loadtxt(Path(args.dataset_dir + args.sequence + "/cam.txt")).astype(np.float32)
            # print(camera_matrix)
            rvec = np.array([0., 0., 0])
            tvec = np.array([0., 0., 0])
            rvec, _ = cv2.Rodrigues(rvec)
            NUM_SHOW_POINTS = 70

            # homogeneous point [x, y, z, w] corresponds to the three-dimensional point [x/w, y/w, z/w].
            project_points = np.array([[0, 1.7, 3, 1]]).reshape(1, 1, 4)
            # homogeneous point [x, y, z, w] corresponds to the three-dimensional point [x/w, y/w, z/w].
            project_points_l = np.array([[-0.8, 1.7, 3, 1]]).reshape(1, 1, 4)
            # homogeneous point [x, y, z, w] corresponds to the three-dimensional point [x/w, y/w, z/w].
            project_points_r = np.array([[+0.8, 1.7, 3, 1]]).reshape(1, 1, 4)


            path = args.dataset_dir + args.sequence + "/frame{0:06d}.png"

            df = pd.read_csv(args.dataset_dir.replace('_frames', '') + 'info/{}-0-static-deleted.csv'.format(seq), sep=",")
            print(df)

            global_pose = np.identity(4)
            poses = [global_pose[0:3, :].reshape(1, 12)]

            for i in range(len(df)):
                v = df['linear_speed'][i]
                dt = 0.1
                r = get_radius(df['real_steer_angle'][i] / WHEEL_STEER_RATIO)
                alpha = v * dt / r
                rot = [0, alpha, 0]
                px, py = rotate_point(0, 0, alpha, -r, 0)
                trans = [px + r, 0, py]
                pose = torch.tensor(trans + rot).reshape(1, 6)
                pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
                pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
                global_pose = global_pose @ np.linalg.inv(pose_mat)
                poses.append(global_pose[0:3, :].reshape(1, 12))

            n = len(poses)
            poses = np.array(poses).reshape(n, 3, 4)

            x = np.zeros((n, 1, 4))
            x[:, :, -1] = 1
            poses = np.concatenate([poses, x], axis=1)
                

            for i in tqdm(range(n - 1)):
                crt_pose = np.stack(inv(poses[i]).dot(x) for x in poses[i:])

                world_points = project_points.dot(crt_pose.transpose((0, 2, 1)))[0, 0]
                world_points_l = project_points_l.dot(crt_pose.transpose((0, 2, 1)))[0, 0]
                world_points_r = project_points_r.dot(crt_pose.transpose((0, 2, 1)))[0, 0]

                show_img = cv2.imread(path.format(i)).astype(np.float32) / 255.

                world_points_show = np.concatenate([
                    world_points[:NUM_SHOW_POINTS][:, :3],
                    world_points_l[:NUM_SHOW_POINTS][:, :3],
                    world_points_r[:NUM_SHOW_POINTS][:, :3]
                ])

                rvec2 = np.eye(3)  # it is almost the identity matrix
                show_points = cv2.projectPoints(world_points_show.astype(np.float64), rvec2, tvec,
                                                camera_matrix, None)[0]
                show_points_l = cv2.projectPoints(world_points_l[:NUM_SHOW_POINTS][:, :3].astype(np.float64), rvec2, tvec,
                                                  camera_matrix, None)[0]
                show_points_r = cv2.projectPoints(world_points_r[:NUM_SHOW_POINTS][:, :3].astype(np.float64), rvec2, tvec,
                                                  camera_matrix, None)[0]
                show_points = show_points.astype(np.int)[:, 0]
                show_points_l = show_points_l.astype(np.int)[:, 0]
                show_points_r = show_points_r.astype(np.int)[:, 0]
                overlay = np.zeros_like(show_img)
                overlay_limited = np.zeros_like(show_img)
                # overlay[:, :, 0] = 255

                ok = True

                # cv2.imshow('img', show_img)  # distances / distances.max())
                # cv2.waitKey(0)

                for it, p1, p2, p3, p4 in zip(range(len(show_points_l) - 1), show_points_l[:-1], show_points_r[:-1],
                                              show_points_l[1:], show_points_r[1:]):
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    x4, y4 = p4
                    pts = np.array([(x1, y1), (x3, y3), (x4, y4), (x2, y2)])

                    overlay = cv2.drawContours(overlay, [pts], 0, (0, 255, 0), cv2.FILLED)

                alpha = 1.0
                show_img_og = np.copy(show_img)
                # show_img = cv2.addWeighted(overlay_limited, alpha, show_img, 1, 0)
                show_img_og = cv2.addWeighted(overlay, alpha, show_img_og, 1, 0)

                if np.sum(overlay) > 0.0:
                	pass
                    # cv2.imwrite(args.segmentation_path + "labels/GTLabels/" + path.format(i).replace('/', '\\'), distances)
                    
                cv2.imshow('res', show_img_og)
                cv2.waitKey(0)
                # plt.waitforbuttonpress()
                # plt.close()


if __name__ == '__main__':
    main()
