# Finetune seg on RSNA
CUDA_VISIBLE_DEVICES=1 python train.py --name efmp --stage train --model vit_base_patch16 --task RSNA --img_size 224 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
    --output_dir "output/RSNA/10/" --data_volume '10' --num_steps 1000  --eval_batch_size 512 \
    --learning_rate 5e-4 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 256 --weight_decay 0.05


