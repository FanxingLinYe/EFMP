# MED1:
# CUDA_VISIBLE_DEVICES=5 python3 RSNA_FTLP.py --name efmp --stage test --model vit_base_patch16 --task RSNA --num_classes 2 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/Fine-tuning/RSNA/output/RSNA/1/efmp_bestauc_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
#     --output_dir "output/RSNA/1/" --fp16 --fp16_opt_level O2 \
#     --data_volume 1 --eval_batch_size 512 --img_size 224
# MED10:
# CUDA_VISIBLE_DEVICES=5 python3 RSNA_FTLP.py --name efmp --stage test --model vit_base_patch16 --task RSNA --num_classes 2 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/Fine-tuning/RSNA/output/RSNA/10/efmp_bestauc_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
#     --output_dir "output/RSNA/10/" --fp16 --fp16_opt_level O2 \
#     --data_volume 10 --eval_batch_size 512 --img_size 224
# MED100:
CUDA_VISIBLE_DEVICES=5 python3 RSNA_FTLP.py --name efmp --stage test --model vit_base_patch16 --task RSNA --num_classes 2 \
    --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/Fine-tuning/RSNA/output/RSNA/100/efmp_bestauc_checkpoint.bin" \
    --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
    --output_dir "output/RSNA/100/" --fp16 --fp16_opt_level O2 \
    --data_volume 100 --eval_batch_size 512 --img_size 224