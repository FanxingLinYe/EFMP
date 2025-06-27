# Finetune seg on RSNA
CUDA_VISIBLE_DEVICES=6 python train.py --name efmp --stage train --model vit_base_patch16 --task RSNA --img_size 224 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
    --output_dir "output/RSNA/100/" --data_volume '100' --num_steps 1000  --eval_batch_size 512 \
    --learning_rate 3e-3 --warmup_steps 100 --fp16 --fp16_opt_level O2 --train_batch_size 512 --weight_decay 0.05