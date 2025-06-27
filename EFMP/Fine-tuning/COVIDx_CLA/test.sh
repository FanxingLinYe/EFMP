# MED1:
# CUDA_VISIBLE_DEVICES=3 python3 COVIDx_FTLP.py --name ecamp --stage test --model vit_base_patch16 --task COVIDx --num_classes 3 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/ECAMP/Fine-tuning/COVIDx_CLA/output/COVIDx/1/ecamp_bestauc_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/COVIDX-7/' \
#     --output_dir "output/COVIDx/1/" --fp16 --fp16_opt_level O2 \
#     --data_volume 1 --eval_batch_size 512 --img_size 224
# MED10:
# CUDA_VISIBLE_DEVICES=3 python3 COVIDx_FTLP.py --name ecamp --stage test --model vit_base_patch16 --task COVIDx --num_classes 3 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/ECAMP/Fine-tuning/COVIDx_CLA/output/COVIDx/10/ecamp_bestauc_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/COVIDX-7/' \
#     --output_dir "output/COVIDx/10/" --fp16 --fp16_opt_level O2 \
#     --data_volume 10 --eval_batch_size 512 --img_size 224
# MED100:
# CUDA_VISIBLE_DEVICES=3 python3 COVIDx_FTLP.py --name ecamp --stage test --model vit_base_patch16 --task COVIDx --num_classes 3 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/ECAMP/Fine-tuning/COVIDx_CLA/output/COVIDx/100/ecamp_bestauc_checkpoint.bin" \
#     --dataset_path '/mnt/data0/YXG/COVIDX-7/' \
#     --output_dir "output/COVIDx/100/" --fp16 --fp16_opt_level O2 \
#     --data_volume 100 --eval_batch_size 512 --img_size 224
