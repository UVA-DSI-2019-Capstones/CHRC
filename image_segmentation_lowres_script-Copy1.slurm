#!/bin/bash
#SBATCH --mem=16GB
#SBATCH --time=40:00:00
#SBATCH --output=/sfs/qumulo/qhome/kk4ze/segement_output.out
#SBATCH --error=/sfs/qumulo/qhome/kk4ze/segmentation.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kk4ze@virginia.edu
#SBATCH --account=DSone 
#SBATCH --partition=standard
#SBATCH --nodes=2


cd  /scratch/kk4ze/
module load openslide-python/1.1.1-py3
module load anaconda3
python ./seg_image_low_res_2200x2200.py
