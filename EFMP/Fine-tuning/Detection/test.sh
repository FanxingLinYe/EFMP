# MED1:
# CUDA_VISIBLE_DEVICES=0 python3 train.py --name ecamp --stage test --model vit_base_patch16 --task RSNA \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/ECAMP/Fine-tuning/Detection/output/RSNA/ECAMP/1/ecamp_bestmap_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
#     --output_dir "output/RSNA/ECAMP/1/" --fp16 --fp16_opt_level O2 \
#     --data_volume 1 --eval_batch_size 512 --img_size 224
# MED10:
# CUDA_VISIBLE_DEVICES=0 python3 train.py --name ecamp --stage test --model vit_base_patch16 --task RSNA \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/ECAMP/Fine-tuning/Detection/output/RSNA/ECAMP/10/ecamp_bestmap_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
#     --output_dir "output/RSNA/ECAMP/10/" --fp16 --fp16_opt_level O2 \
#     --data_volume 10 --eval_batch_size 512 --img_size 224
# MED100:
# CUDA_VISIBLE_DEVICES=0 python3 train.py --name ecamp --stage test --model vit_base_patch16 --task RSNA \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/ECAMP/Fine-tuning/Detection/output/RSNA/ECAMP/100/ecamp_bestmap_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
#     --output_dir "output/RSNA/ECAMP/100/" --fp16 --fp16_opt_level O2 \
#     --data_volume 100 --eval_batch_size 512 --img_size 224
