# COVIDx
# For 216:
# CUDA_VISIBLE_DEVICES=5 python COVIDx_FTLP.py --name ecamp --stage train --model vit_base_patch16 --task COVIDx --num_classes 3 \
#     --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/COVIDX-7/' \
#     --output_dir "output/COVIDx/100/" --data_volume '100' --num_steps 30000  --eval_batch_size 512 --img_size 224 \
#     --learning_rate 1e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 768

CUDA_VISIBLE_DEVICES=2 python COVIDx_FTLP.py --name ecamp --stage train --model vit_base_patch16 --task COVIDx --num_classes 3 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/COVIDX-7/' \
    --output_dir "output/COVIDx/100/" --data_volume '100' --num_steps 30000  --eval_batch_size 256 --img_size 224 \
    --learning_rate 2e-2 --warmup_steps 500 --fp16 --fp16_opt_level O2 --train_batch_size 256 --gradient_accumulation_steps 3

