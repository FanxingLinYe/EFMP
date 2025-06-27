# 数据量 1%
CUDA_VISIBLE_DEVICES=0 python train.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '$PATH TO efmp_ViT_Base_16.pth' --dataset_path '$PATH TO CheXpert' \
    --output_dir "output/CheXpert/1/" --data_volume '1' --num_steps 30000 --eval_batch_size 1024 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 768 --mode Finetune

# 数据量 10%
CUDA_VISIBLE_DEVICES=0 python train.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '$PATH TO efmp_ViT_Base_16.pth' --dataset_path '$PATH TO CheXpert' \
    --output_dir "output/CheXpert/10/" --data_volume '10' --num_steps 90000 --eval_batch_size 1024 --img_size 224 \
    --learning_rate 5e-3 --warmup_steps 1500 --fp16 --fp16_opt_level O2 --train_batch_size 768 --mode Finetune

# 数据量 100%
CUDA_VISIBLE_DEVICES=0 python train.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '$PATH TO efmp_ViT_Base_16.pth' --dataset_path '$PATH TO CheXpert' \
    --output_dir "output/CheXpert/100/" --data_volume '100' --num_steps 90000 --eval_batch_size 1024 --img_size 224 \
    --learning_rate 4e-3 --warmup_steps 1500 --fp16 --fp16_opt_level O2 --train_batch_size 768 --mode Finetune
