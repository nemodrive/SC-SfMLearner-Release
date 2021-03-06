# Contains a class for loading the pose estimation model
import torch
import numpy as np
from numpy.linalg import inv
from imageio import imread
from skimage.transform import resize as imresize
from utils import read_calib_file
from inverse_warp import pose_vec2mat
import cv2
import pandas as pd
import os
import models
from tqdm import tqdm


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class PoseComposition:
    def __init__(self, img_height, img_width):
        self.pose_net = None
        self.camera_matrix = None
        self.img_height = img_height
        self.img_width = img_width

    def load_model(self, path):
        self.pose_net = models.PoseNet().to(device)
        weights_pose = torch.load(path)
        self.pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
        self.pose_net.eval()

    def load_camera_matrix(self, path, sample_image):
        sample_height, sample_width = sample_image.shape[0], sample_image.shape[1]
        zoom_y = self.img_height / sample_height
        zoom_x = self.img_width / sample_width
        self.camera_matrix = read_calib_file(2, path, zoom_x, zoom_y)

    def load_pose(self, path):
        pose_df = pd.read_csv(path, sep=" ", header=None)
        return pose_df

    '''
    Inputs:
        img: original image
        crop_left: percentage to crop on the left side of the image
        crop_right: percentage to crop on the right side of the image
        crop_bot: percentage to crop on the bottom side of the image
        crop_top: percentage to crop on the top side of the image
    Outputs:
        proc_img: processed image
    '''
    def preprocess_img(self, img, crop_left=0, crop_right=0, crop_bot=0, crop_top=0):
        proc_img = imresize(img, (self.img_height, self.img_width)).astype(np.float32)

        proc_img = np.transpose(proc_img, (2, 0, 1))

        img_height, img_width, _ = proc_img.shape

        new_top = int(crop_top * img_height)
        new_bot = img_height - int(crop_bot * img_height)
        new_left = int(crop_left * img_width)
        new_right = img_width - int(crop_right * img_width)

        proc_img = proc_img[new_top:new_bot, new_left:new_right]
        proc_img = ((torch.from_numpy(proc_img).unsqueeze(0) / 255. - 0.5) / 0.5).to(device)

        return proc_img

    '''
    Inputs:
        frames: list of frames to apply the pose estimation on
        translation_scale: scale factor for the translation
    Outputs:
        world_points: array of positions of the center of the car
        world_points_l: array of positions of the center of the left wheel of the car
        world_points_r: array of positions of the center of the right wheel of the car
    '''
    def compose_transforms(self, frames, translation_scale):
        print(self.camera_matrix)
        global_pose = np.identity(4)
        poses = [global_pose[0:3, :].reshape(1, 12)]

        img1 = frames[0]
        tensor_img1 = self.preprocess_img(img1)

        for i in tqdm(range(len(frames) - 1)):
            img2 = frames[i + 1]
            tensor_img2 = self.preprocess_img(img2)

            pose = self.pose_net(tensor_img1, tensor_img2)
            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().detach().numpy()
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @ np.linalg.inv(pose_mat)

            poses.append(global_pose[0:3, :].reshape(1, 12))

            # update
            tensor_img1 = tensor_img2

        transforms = np.concatenate(poses, axis=0)

        #concatenate all transforms into a np array
        pose = np.array(transforms).reshape((len(transforms), 3, 4))

        x = np.zeros((len(pose), 1, 4))
        x[:, :, -1] = 1

        # add the last row to the pose arrays
        pose = np.concatenate([pose, x], axis=1)

        # scale the translations
        pose[:, :, -1] = translation_scale * pose[:, :, -1]

        # this is with the ground truth pose
        # df_gt = pd.read_csv("/HDD1_2TB/storage/KITTI/data_odometry_color/dataset/poses/09.txt", sep=" ", header=None)
        # pose_gt = df_gt.values.reshape((len(df_gt), 3, 4))
        #
        # x_gt = np.zeros((len(pose_gt), 1, 4))
        #
        # x_gt[:, :, -1] = 1
        #
        # pose_gt = np.concatenate([pose_gt, x_gt], axis=1)

        # compute the relative pose between the first pose and every following one
        crt_pose = np.stack(inv(pose[0]).dot(x) for x in pose[0:])

        rvec = np.array([0., 0., 0])
        tvec = np.array([0., 0., 0])
        rvec, _ = cv2.Rodrigues(rvec)

        # the _l and _r points correspond to the left and right wheels
        # homogeneous point [x, y, z, w] corresponds to the three-dimensional point [x/w, y/w, z/w].
        project_points = np.array([[0, 1.7, 3, 1]]).reshape(1, 1, 4)
        # homogeneous point [x, y, z, w] corresponds to the three-dimensional point [x/w, y/w, z/w].
        project_points_l = np.array([[-0.8, 1.7, 3, 1]]).reshape(1, 1, 4)
        # homogeneous point [x, y, z, w] corresponds to the three-dimensional point [x/w, y/w, z/w].
        project_points_r = np.array([[+0.8, 1.7, 3, 1]]).reshape(1, 1, 4)

        world_points = project_points.dot(crt_pose.transpose((0, 2, 1)))[0, 0]
        world_points_l = project_points_l.dot(crt_pose.transpose((0, 2, 1)))[0, 0]
        world_points_r = project_points_r.dot(crt_pose.transpose((0, 2, 1)))[0, 0]

        # show points are for displaying the world points on a frame
        world_points_show = np.concatenate([
            world_points[:, :3],
            world_points_l[:, :3],
            world_points_r[:, :3]
        ])

        rvec2 = crt_pose[0][:3, :3]  # it is almost the identity matrix
        show_points = cv2.projectPoints(world_points_show.astype(np.float64), rvec2, tvec,
                                        self.camera_matrix, None)[0]
        show_points_l = cv2.projectPoints(world_points_l[:, :3].astype(np.float64), rvec2, tvec,
                                          self.camera_matrix, None)[0]
        show_points_r = cv2.projectPoints(world_points_r[:, :3].astype(np.float64), rvec2, tvec,
                                          self.camera_matrix, None)[0]

        show_points = show_points.astype(np.int)[:, 0]
        show_points_l = show_points_l.astype(np.int)[:, 0]
        show_points_r = show_points_r.astype(np.int)[:, 0]

        img1 = imresize(frames[0], (self.img_height, self.img_width)).astype(np.float32)
        overlay = np.zeros_like(img1)

        for it, p1, p2, p3, p4 in zip(range(len(show_points_l) - 1), show_points_l[:-1], show_points_r[:-1],
                                      show_points_l[1:], show_points_r[1:]):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            pts = np.array([(x1, y1), (x3, y3), (x4, y4), (x2, y2)])

            overlay = cv2.drawContours(overlay, [pts], 0, (0, 255, 0), cv2.FILLED)

        alpha = 1.0
        show_img = cv2.addWeighted(overlay, alpha, img1 / 255.0, 1, 0)

        cv2.imshow('path', show_img)
        cv2.waitKey(0)

        return world_points, world_points_l, world_points_r


if __name__ == "__main__":
    frames_dir = '/HDD1_2TB/storage/KITTI/data_odometry_color/dataset/sequences/09/image_2/'
    frame_names = sorted(os.listdir(frames_dir))

    frames = [imread(frames_dir + f).astype(np.float32) for f in frame_names[:50]]

    compose = PoseComposition(256, 832)
    compose.load_camera_matrix('/HDD1_2TB/storage/KITTI/data_odometry_color/dataset/sequences/09/calib.txt', frames[0])
    compose.load_model('pretrained_models/NeurIPS_Models/pose/cs+k_pose.tar')
    compose.compose_transforms(frames, 53)

    print(frame_names)