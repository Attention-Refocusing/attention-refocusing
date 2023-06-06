#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16000
#SBATCH --exclude=gpu-b10-3
source ~/.bashrc
cd /home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/demo
source activate detectron
module load cuda/11.6.2

# python demo.py --config-file '/home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml' \
# --input '/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/color_img/*' \
# --output  True_stable_self_cross \
# --opts MODEL.WEIGHTS /home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/demo/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth

python demo.py --config-file '/home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml' \
--input '/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss_updated/color//*' \
--output  Attend-and-Excite \
--opts MODEL.WEIGHTS /home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/demo/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth

# python demo.py --config-file '/home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml' \
# --input '/home/quynhpt/scratch/code/Gligen_v2/True_self_cross/color_img/*' \
# --output  True_self_cross \
# --opts MODEL.WEIGHTS /home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/demo/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth

# python demo.py --config-file '/home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml' \
# --input '/home/quynhpt/scratch/code/Gligen_v2/True_only_self/color_img/*' \
# --output  True_only_self \
# --opts MODEL.WEIGHTS /home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/demo/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth

# python demo.py --config-file '/home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml' \
# --input '/home/quynhpt/scratch/code/layout-guidance/hrs_save/color/*.png' \
# --output  layout-guidance \
# --opts MODEL.WEIGHTS /home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/demo/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth

# python demo.py --config-file '/home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml' \
# --input '/home/quynhpt/scratch/code/multi_diffu/HRS_loss/color/*' \
# --output  multi_diffu \
# --opts MODEL.WEIGHTS /home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/demo/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth


# python demo.py --config-file '/home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml' \
# --input '/home/quynhpt/scratch/code/Gligen_original/True_stable_loss/color_img/*' \
# --output stable_loss \
# --opts MODEL.WEIGHTS /home/quynhpt/scratch/code/HRS_benchmark/codes/eval_metrics/colors/MaskDINO/demo/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth
