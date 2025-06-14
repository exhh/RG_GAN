run=1
seed=123

python train.py \
    --name maps_rggan \
    --dataset_mode unaligneds1 \
    --model rg_gan_s1 \
    --no_html \
    --dataroot \ lesion_detection/DA \
    --input_nc 1 \
    --output_nc 1 \
    --gpu_ids 0 \
    --batch_size 1 \
    --load_size 286 \
    --crop_size 256 \
    --n_epochs 35 \
    --n_epochs_decay 0 \
    --run_number ${run} \
    --seed ${seed}



python train.py \
    --name maps_rggan \
    --model rg_gan_s2 \
    --no_html \
    --dataroot \ lesion_detection/DA \
    --input_nc 1 \
    --output_nc 1 \
    --gpu_ids 0 \
    --batch_size 1 \
    --continue_train \
    --epoch 35 \
    --epoch_count 36 \
    --load_size 286 \
    --crop_size 256 \
    --n_epochs 60 \
    --n_epochs_decay 0 \
    --no_flip \
    --run_number ${run} \
    --seed ${seed}


cp checkpoints/maps_rggan/best_EMA_net_D_A.pth checkpoints/maps_rggan/bests2_EMA_net_D_A.pth
cp checkpoints/maps_rggan/best_EMA_net_D_B.pth checkpoints/maps_rggan/bests2_EMA_net_D_B.pth
cp checkpoints/maps_rggan/best_EMA_net_G_A.pth checkpoints/maps_rggan/bests2_EMA_net_G_A.pth
cp checkpoints/maps_rggan/best_EMA_net_G_B.pth checkpoints/maps_rggan/bests2_EMA_net_G_B.pth
cp checkpoints/maps_rggan/best_EMA_net_LD_GA.pth checkpoints/maps_rggan/bests2_EMA_net_LD_GA.pth

cp checkpoints/maps_rggan/best_net_D_A.pth checkpoints/maps_rggan/bests2_net_D_A.pth
cp checkpoints/maps_rggan/best_net_D_B.pth checkpoints/maps_rggan/bests2_net_D_B.pth
cp checkpoints/maps_rggan/best_net_G_A.pth checkpoints/maps_rggan/bests2_net_G_A.pth
cp checkpoints/maps_rggan/best_net_G_B.pth checkpoints/maps_rggan/bests2_net_G_B.pth
cp checkpoints/maps_rggan/best_net_LD_GA.pth checkpoints/maps_rggan/bests2_net_LD_GA.pth


python train.py \
    --name maps_rggan \
    --model rg_gan \
    --no_html \
    --dataroot \ lesion_detection/DA \
    --input_nc 1 \
    --output_nc 1 \
    --gpu_ids 0 \
    --batch_size 1 \
    --continue_train \
    --epoch bests2_EMA \
    --epoch_count 61 \
    --load_size 286 \
    --crop_size 256 \
    --n_epochs 80 \
    --n_epochs_decay 0 \
    --no_flip \
    --run_number ${run} \
    --seed ${seed}
