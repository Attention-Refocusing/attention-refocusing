#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16000
#SBATCH --exclude=gpu-b10-3
source ~/.bashrc
cd /home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master
source activate detectron
module load cuda/11.6.2


# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/drawbench/tru_counting/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/Attend-and-Excite/drawbench_counting.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
	--input "/home/quynhpt/scratch/code/Attend-and-Excite/drawbench/tru_counting_v2/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/Attend-and-Excite/drawbench_counting_v2.pkl" \
	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
	--input "/home/quynhpt/scratch/code/Attend-and-Excite/drawbench/tru_spa_v2/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/Attend-and-Excite/drawbench_spatial_v2.pkl" \
	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/drawbench_true_gli/spatial_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/GLIGEN_self_cross_gate_true/drawbench_spatial.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"




# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/layout-guidance/hrs_save/counting_0_500/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/layout_loss/counting_500.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/multi_diffu/HRS_save_loss/spatial/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/multi_diffu/spatial.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# # python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# # 	--input "/home/quynhpt/scratch/code/multi_diffu/hrs_save/spatial_1000/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/multi_diffu/spatial_1000.pkl" \
# # 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/multi_diffu/HRS_save_loss/counting_0_and_3000/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/multi_diffu/counting_500.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/multi_diffu/HRS_save_loss/counting_500_1499/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/multi_diffu/counting_500_1499.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/counting_1500_2499_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/True_stable_self_cross/counting_1500_2499.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_only_cross/size_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/True_only_cross/size.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
	# --input "/home/quynhpt/scratch/code/multi_diffu/drawbench/counting/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/multi_diffu/drawbench_counting.pkl" \
	# --output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss_updated/counting_500/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/Attend-and-Excite/counting_5000.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss_updated/counting_500_1499/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/Attend-and-Excite/counting_500_1499.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss_updated/counting_1500_2499/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/Attend-and-Excite/counting_1500_2499.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss_updated/counting_2500_3000/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/Attend-and-Excite/counting_2500_3000.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss_updated/spatial_500/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/Attend-and-Excite/spatial_500.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss_updated/spatial_1000/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/Attend-and-Excite/spatial_1000.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss_updated/size/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/Attend-and-Excite/size.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"




# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/layout-guidance/drawbench/counting_true/*.png" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/layout_loss/drawbench_counting.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/layout-guidance/drawbench/spatial/*.png" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/layout_loss/drawbench_spatial.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/drawbench/counting/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/att_exc/drawbench_counting.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/drawbench/cospatial/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/att_exc/drawbench_spatial.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# # python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/drawbench_truest/counting_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/True_stable_self_cross/drawbench_counting.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/drawbench_truest/spatial_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/True_stable_self_cross/drawbench_spatial.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"


# /home/quynhpt/scratch/code/Gligen_v2/True_only_cross/size_img
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/counting_1500_2499_img" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/True_stable_self_cross/counting_2500_3000.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/counting_500_1499_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/True_stable_self_cross/counting_500_1499.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/counting_500_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/True_stable_self_cross/counting_500.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"


# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss/counting_2500_3000/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/att_exc/counting_2500_3000.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss/counting_1500_2499/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/att_exc/counting_1500_2499.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss/counting_500_1499/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/att_exc/counting_500_1499.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss/counting_500/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/att_exc/counting_500.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Attend-and-Excite/hrs_loss/size/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/att_exc/size.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/spatial_500_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/True_stable_self_cross/spatial_500.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/spatial_1000_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/True_stable_self_cross/spatial_1000.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/size_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/True_stable_self_cross/size.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"


# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/layout-guidance/hrs_save/spatial_1000/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/layout_loss/spatial_1000.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/layout-guidance/hrs_save/spatial_500/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/layout_loss/spatial_500.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/layout-guidance/hrs_save/size/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/compositions/layout_loss/size.pkl" \
	# --output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"

# /home/quynhpt/scratch/code/multi_diffu/HRS_loss/color
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/counting_500_1499_img/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/layout_loss/counting_2500_3000.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/counting_500/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/layout_loss/counting_500.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/counting_1500_2499/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/layout_loss/counting_1500_2499.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"
# python demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
# 	--input "/home/quynhpt/scratch/code/Gligen_v2/True_stable_self_cross/counting_500_1499/*" --pkl_pth "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting/layout_loss/counting_500_1499.pkl" \
# 	--output "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/visual_box_GLIGEN_counting_cross_gate09_06" --opts MODEL.WEIGHTS "/home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/detection/UniDet-master/checkpoint/Partitioned_COI_RS101_2x.pth"


