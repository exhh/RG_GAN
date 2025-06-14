for ep in best best_EMA bests2 bests2_EMA
do
    cp checkpoints/maps_rggan/${ep}_net_LD_GA.pth checkpoints/eval/

    python eval_syn_real.py --datadir lesion_detection/datasets \
      --datasetname liver \
      --eval_result_folder lesion_detection/experiments_ganLD \
      --train_val_test test \
      --filtering True \
      --fix_test True \
      --test_all True \
      --name eval \
      --model eval \
      --dataroot lesion_detection/experiments_ganLD \
      --input_nc 1 \
      --output_nc 1  \
      --gpu_ids 0 \
      --no_flip \
      --dataset_mode test \
      --model_suffix "_GA" \
      --epoch ${ep} \
      --run_number 1
done
