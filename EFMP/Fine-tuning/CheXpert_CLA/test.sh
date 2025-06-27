# MED1:
# CUDA_VISIBLE_DEVICES=5 python3 CheXpert_FTLP.py --name efmp --stage test --model vit_base_patch16 --task CheXpert --num_classes 5 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/Fine-tuning/CheXpert_CLA/output/CheXpert/1/efmp_bestauc_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/CheXpert/data/' \
#     --output_dir "output/CheXpert/1/" --fp16 --fp16_opt_level O2 \
#     --data_volume 1 --eval_batch_size 512 --img_size 224
# MED10:
# CUDA_VISIBLE_DEVICES=0 python3 CheXpert_FTLP.py --name efmp --stage test --model vit_base_patch16 --task CheXpert --num_classes 5 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/Fine-tuning/CheXpert_CLA/output/CheXpert/10/efmp_bestauc_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/CheXpert/data/' \
#     --output_dir "output/CheXpert/10/" --fp16 --fp16_opt_level O2 \
#     --data_volume 10 --eval_batch_size 512 --img_size 224
# MED100:
CUDA_VISIBLE_DEVICES=1 python3 CheXpert_FTLP.py --name efmp --stage test --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/Fine-tuning/CheXpert_CLA/output/CheXpert/100/efmp_bestauc_checkpoint.bin" \
    --dataset_path '/mnt/data0/YXG/CheXpert/data/' \
    --output_dir "output/CheXpert/100/" --fp16 --fp16_opt_level O2 \
    --data_volume 100 --eval_batch_size 512 --img_size 224