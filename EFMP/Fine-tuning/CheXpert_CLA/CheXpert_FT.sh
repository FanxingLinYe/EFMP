# 数据量 1%
CUDA_VISIBLE_DEVICES=5 python CheXpert_FTLP.py.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/CheXpert/data/' \
    --output_dir "output/CheXpert/1/" --data_volume '1' --num_steps 2000 --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 150 --fp16 --fp16_opt_level O2 --train_batch_size 96 --mode Finetune

# 数据量 10%
CUDA_VISIBLE_DEVICES=6 python CheXpert_FTLP.py.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/CheXpert/data/' \
    --output_dir "output/CheXpert/10/" --data_volume '10' --num_steps 60000 --eval_batch_size 512 --img_size 224 \
    --learning_rate 5e-4 --warmup_steps 1500 --fp16 --fp16_opt_level O2 --train_batch_size 96 --mode Finetune

# 数据量 100%
CUDA_VISIBLE_DEVICES=7 python CheXpert_FTLP.py.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/CheXpert/data/' \
    --output_dir "output/CheXpert/100/" --data_volume '100' --num_steps 200000 --eval_batch_size 512 --img_size 224 \
    --learning_rate 5e-4 --warmup_steps 15000 --fp16 --fp16_opt_level O2 --train_batch_size 96 --mode Finetune

