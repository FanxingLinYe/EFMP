
# For 216:
CUDA_VISIBLE_DEVICES=7 python CheXpert_FTLP.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/CheXpert/data/' \
    --output_dir "output/CheXpert/100/" --data_volume '100' --num_steps 200000 --eval_batch_size 512 --img_size 224 \
    --learning_rate 5e-4 --warmup_steps 15000 --fp16 --fp16_opt_level O2 --train_batch_size 96 --mode Finetune