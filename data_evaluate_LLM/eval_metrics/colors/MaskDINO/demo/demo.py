"""
python demo.py --config-file '/home/abdelrem/T2I_benchmark/codes/eval_metrics/colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml' \
--input '/home/abdelrem/t2i_benchmark/data/t2i_out/sd_v1/colors/*.png' \
--output /home/abdelrem/T2I_benchmark/data/colors/output/sd_v1/ \
--opts MODEL.WEIGHTS /home/abdelrem/T2I_benchmark/weights/mask_dino/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.pth
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            os.makedirs(args.output, exist_ok=True)
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    print('save')
                    # os.makedirs(folder_name, exist_ok=True)
                    

                    # import pdb; pdb.set_trace()
                    instances = predictions["instances"].to('cpu')
                    # Eslam: Filter the output based on the confidence-score:
                    mask = predictions["instances"].scores >= 0.5
                    mask = mask.nonzero()
                    mask = mask.to('cpu')
                    img_name = os.path.basename(path).split(".")[0]
                    # loop on predictions:
                    for mask_idx in range(len(mask)):
                        filtered_masks = instances.pred_masks[mask[mask_idx].to('cpu')][0]
                        filtered_classes = instances.pred_classes[mask[mask_idx]]
                        filtered_scores = instances.scores[mask[mask_idx]]
                        out_filename = os.path.join(args.output, img_name+"_mask_"+str(mask_idx)+"_"+str(filtered_classes.cpu().item())+".png")
                        cv2.imwrite(out_filename, filtered_masks.cpu().numpy()*255)
