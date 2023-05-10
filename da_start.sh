#!/bin/bash -l
#SBATCH -J MARL
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=dorian.joubaud@uni.lu
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 4
#SBATCH --time=1-23:59:00
#SBATCH --gpus=2
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --array=0-4
echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"
# Run your iapplication as a job step,  passing its unique array id
# (based on which varying processing can be done)
#module load lang/Anaconda3/2020.02

DA=(ROS Jitter TW SMOTE ADASYN)
#DS=(SyntheticControl Computers HouseTwenty GestureMidAirD3 Chinatown GestureMidAirD2 BeetleFly AllGestureWiimoteY PigAirwayPressure ShapesAll)
conda activate da
#python main.py Wafer MLP ${DA[$SLURM_ARRAY_TASK_ID]} 2> ${DA[$SLURM_ARRAY_TASK_ID]}_error.log
python main_100.py Computers LSTM  $DA[$SLURM_ARRAY_TASK_ID] 0 1 10  2> BC_$DA[$SLURM_ARRAY_TASK_ID].log

