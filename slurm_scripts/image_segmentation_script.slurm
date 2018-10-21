#!/bin/bash
#SBATCH --mem=16GB
#SBATCH --time=40:00:00
#SBATCH --output=/sfs/qumulo/qhome/ss4yd/segement_output.out
#SBATCH --error=/sfs/qumulo/qhome/ss4yd/segmentation.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ss4yd@virginia.edu
#SBATCH --account=DSone 
#SBATCH --partition=standard

cd  /scratch/ss4yd/
module load anaconda3
module load openslide-python/1.1.1-py3
python ./seg_image_1000x1000_512_512.py
