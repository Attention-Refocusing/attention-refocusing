# #!/bin/bash
# #SBATCH -t 2-00:00:00
# #SBATCH --ntasks=1
# #SBATCH -p gpu
# #SBATCH --gres=gpu:a100:1
# #SBATCH --mem=16000
# cd /home/quynhpt/scratch/code/data_evaluate_LLM/code/HRS_benchmark/codes/eval_metrics/counting
# source activate detectron
# module load cuda/11.6.2
python calc_counting_acc.py 'Attend-and-Excite/counting_5000.pkl,Attend-and-Excite/counting_500_1499.pkl' ../../HRS/counting_prompts.csv 1

# python calc_counting_acc.py Attend-and-Excite/drawbench_counting_v2.pkl /home/quynhpt/scratch/code/data_evaluate_LLM/drawbench/label_counting_db.csv 10
