# MED1:
CUDA_VISIBLE_DEVICES=3 python3 train.py --name efmp --stage test --model vit_base_patch16 --model_type ViT-B_16 --num_classes 14 \
    --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/Fine-tuning/NIH_ChestX-ray/output/ChestX-ray14/1/efmp_bestauc_checkpoint.bin" \
    --dataset_path '/mnt/data0/YXG/NIH_ChestX-ray/' \
    --output_dir "output/NIH_ChestX-ray/1/" --fp16 --fp16_opt_level O2 \
    --data_volume 1 --eval_batch_size 512 --img_size 224
# MED10:
# CUDA_VISIBLE_DEVICES=3 python3 train.py --name efmp --stage test --model vit_base_patch16 --model_type ViT-B_16 --num_classes 14 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/Fine-tuning/NIH_ChestX-ray/output/ChestX-ray14/10/efmp_bestauc_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/NIH_ChestX-ray/' \
#     --output_dir "output/NIH_ChestX-ray/10/" --fp16 --fp16_opt_level O2 \
#     --data_volume 10 --eval_batch_size 512 --img_size 224
# MED100:
# CUDA_VISIBLE_DEVICES=3 python3 train.py --name efmp --stage test --model vit_base_patch16 --model_type ViT-B_16 --num_classes 14 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/Fine-tuning/NIH_ChestX-ray/output/ChestX-ray14/100/efmp_bestauc_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/NIH_ChestX-ray/' \
#     --output_dir "output/NIH_ChestX-ray/100/" --fp16 --fp16_opt_level O2 \
#     --data_volume 100 --eval_batch_size 512 --img_size 224