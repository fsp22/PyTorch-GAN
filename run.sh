#!/bin/bash
set -e

# Get current date and time for the folder name
datetime=$(date +"%Y%m%d_%H%M%S")
folder_name="run_$datetime"

# Run the Python script with the specified parameters
cd implementations/gan
python gan.py \
    --n_epochs 200 \
    --batch_size 64 \
    --lr 0.0002 \
    --b1 0.5 \
    --b2 0.999 \
    --n_cpu 8 \
    --latent_dim 100 \
    --img_size 32 \
    --channels 1 \
    --eval_interval 20 \
    --dataset mnist

# Create the directory with the current date and time
mkdir -p "$folder_name"

# Move the output files to the new directory
mv images/ "$folder_name"/
mv temp/ "$folder_name"/
mv training.log "$folder_name"/
mv *.pth "$folder_name"/

echo "Output files moved to $folder_name/"
