# Finetune DET on RSNA
# ECAMP 1%
CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task RSNA  \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/'\
    --output_dir "output/RSNA/ECAMP/1/" --data_volume '1' --num_steps 3000  --eval_batch_size 1024 \
    --learning_rate 5e-4 --warmup_steps 5 --fp16 --start_eval 60 --train_batch_size 96 --weight_decay 0.05
# ECAMP 10%
# CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task RSNA --img_size 224 \
#     --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
#     --output_dir "output/RSNA/ECAMP/10/" --data_volume '10' --num_steps 3000  --eval_batch_size 1024 \
#     --learning_rate 5e-4 --warmup_steps 5 --fp16 --start_eval 100 --train_batch_size 256 --weight_decay 0.05
# ECAMP 100%
# CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task RSNA --img_size 224 \
#     --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
#     --output_dir "output/RSNA/ECAMP/100/" --data_volume '100' --num_steps 20000  --eval_batch_size 1024 \
#     --learning_rate 5e-4 --warmup_steps 30 --fp16 --start_eval 50 --fp16_opt_level O2 --train_batch_size 1024 --weight_decay 0.05
# CUDA_VISIBLE_DEVICES=0 python train.py --name ecamp --stage train --model vit_base_patch16 --task RSNA --img_size 224 \
#     --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
#     --output_dir "output/RSNA/ECAMP/100/" --data_volume '100' --num_steps 20000  --eval_batch_size 1024 \
#     --learning_rate 2.5e-4 --warmup_steps 30 --fp16 --start_eval 50 --fp16_opt_level O2 --train_batch_size 512 --weight_decay 0.05
