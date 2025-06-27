
# For 216:
CUDA_VISIBLE_DEVICES=6 python CheXpert_FTLP.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/CheXpert/data/' \
    --output_dir "output/CheXpert/10/" --data_volume '10' --num_steps 60000 --eval_batch_size 512 --img_size 224 \
    --learning_rate 5e-4 --warmup_steps 1500 --fp16 --fp16_opt_level O2 --train_batch_size 96 --mode Finetune