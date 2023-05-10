#!/bin/bash -l
#SBATCH -J DA
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=dorian.joubaud@uni.lu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=1-23:59:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --array 0-9




echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"
# Run your application as a job step,  passing its unique array id
# (based on which varying processing can be done)
#module load lang/Anaconda3/2020.02
#VALUES=(Computers HouseTwenty Chinatown BeetleFly ToeSegmentation1 MoteStrain FreezerSmallTrain DodgerWeekend Coffee ShapeletSim)
VALUES=(DodgerLoopGame ToeSegmentation2 BirdChicken SemgHandGenderCh2 FreezerRegularTrain PowerCons)
#3-0 VALUES=(Meat UMD SmoothSubspace ScreenType SmallKitchenAppliances ArrowHead RefrigerationDevices BME LargeKitchenAppliances)
#4-0 VALUES=(EthanolLevel)
#5-0VALUES=(SemgHandSubjectCh2 MixedShapesSmallTrain MixedShapesRegularTrain)
sbatch da.sh ${VALUES[$SLURM_ARRAY_TASK_ID]}
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
~                                                                                            
