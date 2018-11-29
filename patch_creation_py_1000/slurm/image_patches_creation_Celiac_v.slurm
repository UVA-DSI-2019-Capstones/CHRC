#!/bin/bash
#SBATCH --nodes=2
#SBATCH --mem=20GB
#SBATCH --time=40:00:00
#SBATCH --output=/sfs/qumulo/qhome/ss4yd/patch_Celiac_v_output.out
#SBATCH --error=/sfs/qumulo/qhome/ss4yd/patch_Celiac_v.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ss4yd@virginia.edu
#SBATCH --account=DSone 
#SBATCH --partition=standard

cd  /scratch/ss4yd/patch_creation_py_1000	
module load anaconda3
module load openslide-python/1.1.1-py3
python ./create_custom_patches_Celiac_v_1000.py
