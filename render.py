#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import copy # added this line - William Dormer
import cv2 # added this line - William Dormer
import numpy as np # addeed this line - William Dormer
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

# more lines added to make the video rendering work
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

# the following function is from FSGS for rendering videos. We're attempting to import it here. 
def render_video(source_path, model_path, iteration, views, gaussians, pipeline, background, fps=30):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = copy.deepcopy(views[0])

    # if source_path.find('llff') != -1:
    #     render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'))
    # elif source_path.find('360') != -1:
    #     render_poses = generate_ellipse_path(views)

    render_poses = generate_ellipse_path(views) # in our dataset, we always want to generate ellipse path. 

    size = (view.original_image.shape[2], view.original_image.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)
    # final_video = cv2.VideoWriter(os.path.join('/ssd1/zehao/gs_release/video/', str(iteration), model_path.split('/')[-1] + '.mp4'), fourcc, fps, size)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)

        img = torch.clamp(rendering["render"], min=0., max=1.)
        torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        final_video.write(video_img)

    final_video.release()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, disable_xyz_log_activation): 
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, disable_xyz_log_activation=disable_xyz_log_activation)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # more code modified by William Dormer
        if args.video:
            render_video(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTestCameras(),
                         gaussians, pipeline, background, args.fps)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--disable_xyz_log_activation", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # modifications by William Dormer in line with FSGS
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", default=30, type=int)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.disable_xyz_log_activation)