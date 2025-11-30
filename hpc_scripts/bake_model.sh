#!/bin/bash
### --------------- job name ------------------
#BSUB -J ipeo_model

### --------------- queue name ----------------
#BSUB -q gpuv100

### --------------- GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 6
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=2GB]"

### --------------- wall-clock time ---------------
#BSUB -W 23:59

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/ecosystem_mapper/bakery/ipeo_model_%J.out
#BSUB -e /zhome/a2/c/213547/ecosystem_mapper/bakery/ipeo_model_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/hypersight/venv/bin/activate
python3 /zhome/a2/c/213547/ecosystem_mapper/code/train_fusion.py \
	--epochs 120 \
