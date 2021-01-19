# Contains a class for loading the pose estimation model
import torch
import numpy as np
from numpy.linalg import inv
from utils import read_calib_file
import cv2


class PoseComposition:
    def __init__(self, camera_matrix):
        self.model = None
        self.camera_matrix = None

    def load_model(self, path):
        self.model = torch.load(path)

    def load_camera_matrix(self, path):
        self.camera_matrix = read_calib_file(2, path)

    '''
    Inputs:
        img: original image
        crop_left: percentage to crop on the left side of the image
        crop_right: percentage to crop on the right side of the image
        crop_bot: percentage to crop on the bottom side of the image
        crop_top: percentage to crop on the top side of the image
        scale_h: percentage to scale on height
        scale_w: percentage to scale on width
    Outputs:
        proc_img: processed image
    '''
    def preprocess_img(self, img, crop_left, crop_right, crop_bot, crop_top, scale_h, scale_w):
        img_height, img_width, _ = img.shape
        proc_img = cv2.resize(img, fx=scale_h, fy=scale_w)
        new_top = int(crop_top * img_height)
        new_bot = img_height - int(crop_bot * img_height)
        new_left = int(crop_left * img_width)
        new_right = img_width - int(crop_left * img_width)
        proc_img = proc_img[new_top:new_bot, new_left:new_right]
        return proc_img

    '''
    Inputs:
        transforms: list with pose transforms given as either 1D  or 3x4 2D arrays
        translation_scale: scale factor for the translation
    Outputs:
        world_points: array of positions of the center of the car
        world_points_l: array of positions of the center of the left wheel of the car
        world_points_r: array of positions of the center of the right wheel of the car
    '''
    def compose_transforms(self, transforms, translation_scale):
        #concatenate all transforms into a np array
        pose = np.array(transforms).reshape((len(transforms), 3, 4))

        x = np.zeros((len(pose), 1, 4))
        x[:, :, -1] = 1

        # add the last row to the pose arrays
        pose = np.concatenate([pose, x], axis=1)

        # scale the translations
        pose[:, :, -1] = translation_scale * pose[:, :, -1]

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

        return world_points, world_points_l, world_points_r
