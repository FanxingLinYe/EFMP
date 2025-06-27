# ChestX-ray14
# For 216:

CUDA_VISIBLE_DEVICES=5 python CheXpert_FTLP.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/CheXpert/data/' \
    --output_dir "output/CheXpert/1/" --data_volume '1' --num_steps 2000 --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 150 --fp16 --fp16_opt_level O2 --train_batch_size 96 --mode Finetune