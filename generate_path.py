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
from inverse_warp import pose_mat2vec
from tqdm import tqdm
import seaborn as sns
from inverse_warp import inverse_warp2

parser = argparse.ArgumentParser(description='Script for visualizing the path',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", type=str, help="pretrained DispNet path",
                    default='pretrained_models/NeurIPS_Models/cs-pretrain/exp_pose_model_best.pth.tar')
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
parser.add_argument("--dataset-dir", type=str, help="Dataset directory",
                    default='/mnt/storage/workspace/andreim/kitti/data_odometry_color/dataset/sequences/')
parser.add_argument("--scale-translation", default=False,
                    nargs='*', type=bool, help="specifies if the translation is scaled to gt")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'],
                    nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--segmentation-path",
                    default="/mnt/storage/workspace/andreim/nemodrive/kitti_self_supervised_labels/",
                    nargs='*', type=str, help="path to segmentation dataset")


def read_calib_file(cid, filepath, zoom_x, zoom_y):
    with open(filepath, 'r') as f:
        C = f.readlines()

    def parseLine(L, shape):
        data = L.split()
        data = np.array(data[1:]).reshape(shape).astype(np.float32)
        return data

    proj_c2p = parseLine(C[int(cid)], shape=(3, 4))
    calib = proj_c2p[0:3, 0:3]
    calib[0, :] *= zoom_x
    calib[1, :] *= zoom_y

    return calib


def load_tensor_image(filename, device, args):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)
                       ).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = (
            (torch.from_numpy(img).unsqueeze(0) / 255 - 0.5) / 0.5).to(device)
    return tensor_img


def find_dist(thresh, contours, img_height, img_width):
    distances = np.zeros_like(thresh).astype(np.float)
    sigma = 50

    for x in range(img_height):
        for y in range(img_width):
            dist = cv2.pointPolygonTest(contours[-1], (y, x), True)
            is_in = cv2.pointPolygonTest(contours[-1], (y, x), False)
            distances[x][y] = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(- (dist ** 2) / (2 * sigma ** 2))

    return distances


@torch.no_grad()
def main():
    args = parser.parse_args()

    limit_1 = 8
    limit_2 = 28

    for seq in ["{0:02d}".format(i) for i in range(0, 22)]:
        args.sequence = seq

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        #weights_disp = torch.load(args.pretrained_dispnet)
        #disp_net = models.DispResNet().to(device)
        #disp_net.load_state_dict(weights_disp['state_dict'], strict=False)
        #disp_net.eval()

        image_dir = Path(args.dataset_dir + args.sequence + "/image_2/")

        test_files = sum([image_dir.files('*.{}'.format(ext))
                          for ext in args.img_exts], [])
        test_files.sort()
        test_files = test_files
        print('{} files to test'.format(len(test_files)))

        # TODO: rerun this to complete the missing tensors
        # for i in tqdm(range(len(test_files))):
        #     tensor_i = load_tensor_image(test_files[i], device, args)
        #     torch.save(tensor_i, "/HDD1_2TB/storage/kitti_self_supervised_labels/labels/RGBTensors/" + test_files[i].
        #                replace("/", '\\').replace("png", "pt"))
        #     # img_dict[i] = tensor_i.cpu()
        #     depth_i = 1 / disp_net(tensor_i)
        #     # disp_dict[i] = depth_i.cpu()
        #     torch.save(depth_i, "/HDD1_2TB/storage/kitti_self_supervised_labels/labels/DepthTensors/" + test_files[i].
        #                replace("/", '\\').replace("png", "pt"))

        df = pd.read_csv("results/vo/cs+k_pose_kitti/{}.txt".format(args.sequence), sep=" ", header=None)
        path = args.dataset_dir + args.sequence + "/image_2/{0:06d}.png"
        pose = df.values.reshape((len(df), 3, 4))

        x = np.zeros((len(pose), 1, 4))

        x[:, :, -1] = 1

        pose = np.concatenate([pose, x], axis=1)

        #df_gt = pd.read_csv("/mnt/storage/workspace/andreim/kitti/data_odometry_color/dataset/poses/{}.txt"
        #                    .format(args.sequence), sep=" ", header=None)  # change this with real gt if there is
        #pose_gt = df_gt.values.reshape((len(df), 3, 4))

        #x_gt = np.zeros((len(pose_gt), 1, 4))

        #x_gt[:, :, -1] = 1

        #pose_gt = np.concatenate([pose_gt, x], axis=1)

        pose_scaled = pose.copy()

        if args.scale_translation:
            scale_factor = np.sum(pose_gt[:, :, -1] * pose[:, :, -1]) / np.sum(pose[:, :, -1] ** 2)
            pose[:, :, -1] = scale_factor * pose[:, :, -1]
        else:
            pose[:, :, -1] *= 32.8

        #scale_factor = np.sum(pose_gt[:, :, -1] * pose_scaled[:, :, -1]) / np.sum(pose_scaled[:, :, -1] ** 2)
        #pose_scaled[:, :, -1] = scale_factor * pose_scaled[:, :, -1]

        # print(scale_factor)

        first_img = cv2.imread(path.format(0))
        zoom_y = args.img_height / first_img.shape[0]
        zoom_x = args.img_width / first_img.shape[1]
        camera_matrix = read_calib_file(2, Path(args.dataset_dir + args.sequence + "/calib.txt"), zoom_x, zoom_y)
        print(camera_matrix)
        rvec = np.array([0., 0., 0])
        tvec = np.array([0., 0., 0])
        rvec, _ = cv2.Rodrigues(rvec)
        NUM_SHOW_POINTS = 70

        # homogeneous point [x, y, z, w] corresponds to the three-dimensional point [x/w, y/w, z/w].
        project_points = np.array([[0, 1.65, 3, 1]]).reshape(1, 1, 4)
        # homogeneous point [x, y, z, w] corresponds to the three-dimensional point [x/w, y/w, z/w].
        project_points_l = np.array([[-0.8, 1.65, 3, 1]]).reshape(1, 1, 4)
        # homogeneous point [x, y, z, w] corresponds to the three-dimensional point [x/w, y/w, z/w].
        project_points_r = np.array([[+0.8, 1.65, 3, 1]]).reshape(1, 1, 4)

        if seq in ['03', '04', '06', '11']:
            file_name = 'val.txt'
        if seq in ['09', '10', '19']:
            file_name = 'test.txt'
        else:
            file_name = 'train.txt'

        # file_name = 'train+val.txt'

        f = open(args.segmentation_path + 'labels/ImageSets/Segmentation/' + file_name, 'a+')

        for i in tqdm(range(len(pose))):
            crt_pose = np.stack(inv(pose[i]).dot(x) for x in pose[i:])

            valid_frames = []

            # tensor_i = torch.load("/HDD1_2TB/storage/kitti_self_supervised_labels/labels/RGBTensors/" + test_files[i].
            #                       replace("/", '\\').replace("png", "pt"))
            # depth_i = torch.load("/HDD1_2TB/storage/kitti_self_supervised_labels/labels/DepthTensors/" + test_files[i].
            #                      replace("/", '\\').replace("png", "pt"))

            # for it, pose_t in enumerate(crt_pose[:NUM_SHOW_POINTS]):
            #     rot = torch.tensor(pose_mat2vec(torch.tensor(pose_t).unsqueeze(0)))
            #     trans = torch.tensor(pose_t[:3, 3])
            #     pose_t = torch.cat([trans, rot]).unsqueeze(0).float()
            #     tensor_j = torch.load(
            #         "/HDD1_2TB/storage/kitti_self_supervised_labels/labels/RGBTensors/" + test_files[i + it].
            #         replace("/", '\\').replace("png", "pt"))
            #     depth_j = torch.load(
            #         "/HDD1_2TB/storage/kitti_self_supervised_labels/labels/DepthTensors/" + test_files[i + it].
            #         replace("/", '\\').replace("png", "pt"))
            #
            #     _, _, projected_depth, computed_depth = inverse_warp2(tensor_i, depth_j, depth_i, pose_t.cuda(),
            #                                                           torch.tensor(camera_matrix).unsqueeze(0).cuda())
            #
            #     diff_depth = ((computed_depth - projected_depth).abs() /
            #                   (computed_depth + projected_depth).abs()).clamp(0, 1)
            #
            #     res = diff_depth.mean()
            #
            #     if res < 0.8:
            #         valid_frames.append(it)
            #
            # print(len(valid_frames))

            sum_euler = np.zeros(3)
            for p1, p2 in zip(pose[i + 1:i + NUM_SHOW_POINTS], pose[i + 2:i + NUM_SHOW_POINTS + 1]):
                relative_pose = inv(p1).dot(p2)
                relative_pose = relative_pose.reshape((1, 4, 4))
                sum_euler += (pose_mat2vec(relative_pose) * 180 / np.pi)

            world_points = project_points.dot(crt_pose.transpose((0, 2, 1)))[0, 0]
            world_points_l = project_points_l.dot(crt_pose.transpose((0, 2, 1)))[0, 0]
            world_points_r = project_points_r.dot(crt_pose.transpose((0, 2, 1)))[0, 0]

            show_img = cv2.imread(path.format(i))
            show_img = imresize(show_img, (args.img_height, args.img_width)).astype(np.float32)

            world_points_show = np.concatenate([
                world_points[:NUM_SHOW_POINTS][:, :3],
                world_points_l[:NUM_SHOW_POINTS][:, :3],
                world_points_r[:NUM_SHOW_POINTS][:, :3]
            ])

            rvec2 = crt_pose[0][:3, :3]  # it is almost the identity matrix
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

            #cv2.imshow('img', show_img)  # distances / distances.max())
            #cv2.waitKey(0)

            for it, p1, p2, p3, p4 in zip(range(len(show_points_l) - 1), show_points_l[:-1], show_points_r[:-1],
                                          show_points_l[1:], show_points_r[1:]):
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = p3
                x4, y4 = p4
                pts = np.array([(x1, y1), (x3, y3), (x4, y4), (x2, y2)])

                if it in valid_frames or ok:
                    overlay_limited = cv2.drawContours(overlay_limited, [pts], 0, (0, 0, 255), cv2.FILLED)
                else:
                    ok = False

                overlay = cv2.drawContours(overlay, [pts], 0, (0, 255, 0), cv2.FILLED)

            # for it, p1, p2 in zip(range(len(show_points_l) - 1), show_points[:-1], show_points[1:]):
            #     x1, y1 = p1
            #     x2, y2 = p2
            #
            #     overlay = cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

            alpha = 1.0
            show_img_og = np.copy(show_img)
            # show_img = cv2.addWeighted(overlay_limited, alpha, show_img, 1, 0)
            show_img_og = cv2.addWeighted(overlay, alpha, show_img_og, 1, 0)

            # find the contours of the overlay
            # imgray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            # imgray = np.uint8(imgray)
            # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #
            # cv2.drawContours(show_img, contours, -1, (0, 255, 0), 1)

            # if contours:
            #     distances = find_dist(thresh, contours, args.img_height, args.img_width)
            # else:
            #     distances = np.zeros_like(thresh)

            distances = overlay.copy()

            for _ in range(12):
                distances = cv2.GaussianBlur(distances, (9, 9), 5)

            show_img = cv2.addWeighted(distances / distances.max(), alpha, show_img, 1, 0)

            #cv2.imshow('img', distances / distances.max())  # distances / distances.max())
            #cv2.waitKey(0)

            #cv2.imshow('img', show_img)  # distances / distances.max())
            #cv2.waitKey(0)

            #cv2.imshow('img', overlay)  # distances / distances.max())
            #cv2.waitKey(0)

            # print(distances.max(), distances.min())
            #
            # plt.figure(figsize=(8.32, 2.56), dpi=100)
            # plt.gca().set_axis_off()
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
            #                     hspace=0, wspace=0)
            # plt.margins(0, 0)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            #distances = distances[:, :, 1]
            #sns.set(font_scale=3)
            #sns.heatmap(distances / distances.max(), cbar_kws={'orientation': 'horizontal'})#, cbar=False, xticklabels=False, yticklabels=False)
            #plt.waitforbuttonpress()
            #plt.close()

            # cv2.imwrite("/HDD1_2TB/storage/kitti_self_supervised_labels/labels/JPEGImages/" + seq + "_{0:06d}.jpg".format(i), show_img * 255.0)
            # cv2.imshow("img", show_img)
            # cv2.waitKey(0)
            # show_img = Image.fromarray((show_img * 255).astype(np.uintlimit_1))
            # b, g, r = show_img.split()
            # show_img = Image.merge("RGB", (r, g, b))
            # show_img.save("/HDD1_2TB/storage/kitti_self_supervised_labels/labels/JPEGImages/" + seq + str(i) + '.jpg')
            # overlay = Image.fromarray(overlay.astype(np.uintlimit_1)).convert('P')
            # overlay.show()
            # a = np.array(overlay)
            if np.sum(overlay) > 0.0:
                cv2.imwrite(args.segmentation_path + "labels/SoftLabels/" + path.format(i).replace('/', '\\'), distances)
                cv2.imwrite(args.segmentation_path + "labels/HardLabels/" + path.format(i).replace('/', '\\'), overlay)
                # np.save("/HDD1_2TB/storage/kitti_self_supervised_labels/labels/SoftRoad/" + path.format(i).replace('/', '\\').replace('.png', '.npy'), distances)
                sum_euler = sum_euler[1]
                if -limit_1 <= sum_euler <= limit_1:
                    category = 0
                elif -limit_2 <= sum_euler < -limit_1:
                    category = 1
                elif sum_euler < -limit_2:
                    category = 2
                elif limit_2 >= sum_euler > limit_1:
                    category = 3
                else:
                    category = 4
                # print(sum_euler, category, np.sum(overlay))
                f.write(path.format(i) + ',' + str(category) + ',' + str(sum_euler) + '\n')
            # cv2.imshow('res', overlay)
            # cv2.waitKey(0)
            # plt.waitforbuttonpress()
            # plt.close()
        f.close()


if __name__ == '__main__':
    main()
