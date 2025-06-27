# Finetune Seg on SIIM
CUDA_VISIBLE_DEVICES=6 python train.py --name efmp --stage train --model vit_base_patch16 --task SIIM --img_size 224 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/SIIM/' \
    --output_dir "output/SIIM/100/" --data_volume '100' --num_steps 3000  --eval_batch_size 512 \
    --learning_rate 5e-4 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 512 --weight_decay 0.05