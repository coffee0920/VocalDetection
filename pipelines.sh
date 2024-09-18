#!/bin/bash
# shell for pipelines

# ./run_tensorboard.sh ./data/logs
# example SCNN18
# ./run_tf2.sh python train.py \
#     --model SCNN18 \
#     --train_ds_path SCNN-Jamendo-train.h5 \
#     --val_ds_path SCNN-Jamendo-test.h5 \
#     --test_ds_paths SCNN-test-hard.h5 SCNN-FMA-C-1-fixed-test.h5 SCNN-Jamendo-test.h5 \
#     --classes 2 \
#     --sample_size 32000 1 \
#     --epochs 160 \
#     --batch_size 150 \
#     --loss categorical_crossentropy \
#     --optimizer adadelta \
#     --metrics accuracy \
#     --lr 1.0 \
#     --times 10 \
#     --num_gpus 2 \
#     --training

# ./run_tf2.sh python train.py \
#     --model SCNN18 \
#     --train_ds_path SCNN-FMA-C-1-fixed-train.h5 \
#     --val_ds_path SCNN-FMA-C-1-fixed-test.h5 \
#     --test_ds_paths SCNN-Jamendo-test.h5 \
#                     SCNN-FMA-C-1-fixed-test.h5 \
#                     SCNN-FMA-C-2-fixed-test.h5 \
#                     SCNN-KTV-test.h5 \
#                     SCNN-MIR-1k-train.h5 \
#                     SCNN-Instrumental-non-vocal.h5 \
#                     SCNN-A-Cappella-vocal.h5 \
#                     SCNN-Taiwanese-stream-test.h5 \
#                     SCNN-Taiwanese-CD-test.h5 \
#                     SCNN-Chinese-CD-test.h5 \
#                     SCNN-Classical-test.h5 \
#                     SCNN-test-hard.h5 \
#                     SCNN-RWC.h5 \
#     --classes 2 \
#     --sample_size 32000 1 \
#     --epochs 160 \
#     --batch_size 150 \
#     --loss categorical_crossentropy \
#     --optimizer adadelta \
#     --metrics accuracy \
#     --lr 1.0 \
#     --times 21 \
#     --num_gpus 3 \
#     --verbose 1 \
#     --tag sync/FMA-C-1/2021-02-13_11_SCNN18_SCNN-FMA-C-1-fixed-train_h5_2GPU



# for seed in {0..3}
#  do
#     ./run_tf2.sh python train.py \
#         --model SCNN18 \
#         --train_ds_path SCNN-FMA-C-2-fixed-train.h5 \
#         --train_ds_size 4000 \
#         --seed $seed \
#         --val_ds_path SCNN-FMA-C-2-fixed-test.h5 \
#         --test_ds_paths SCNN-Jamendo-test.h5 \
#                         SCNN-FMA-C-1-fixed-test.h5 \
#                         SCNN-FMA-C-2-fixed-test.h5 \
#                         SCNN-KTV-test.h5 \
#                         SCNN-MIR-1k-train.h5 \
#                         SCNN-Instrumental-non-vocal.h5 \
#                         SCNN-A-Cappella-vocal.h5 \
#                         SCNN-Taiwanese-stream-test.h5 \
#                         SCNN-Taiwanese-CD-test.h5 \
#                         SCNN-Chinese-CD-test.h5 \
#                         SCNN-test-hard.h5 \
#         --classes 2 \
#         --sample_size 32000 1 \
#         --epochs 160 \
#         --batch_size 150 \
#         --loss categorical_crossentropy \
#         --optimizer adadelta \
#         --metrics accuracy \
#         --lr 1.0 \
#         --times 21 \
#         --num_gpus 2 \
#         --training
# done

# ./run_tf2.sh python train.py \
#         --model SCNN18 \
#         --train_ds_path SCNN-Jamendo-train.h5 \
#         --val_ds_path SCNN-FMA-C-1-fixed-test.h5 \
#         --test_ds_paths SCNN-Jamendo-test.h5 \
#                         SCNN-Jamendo-train.h5 \
#                         SCNN-FMA-C-1-fixed-test.h5 \
#                         SCNN-FMA-C-2-fixed-test.h5 \
#                         SCNN-KTV-test.h5 \
#                         SCNN-Taiwanese-CD-test.h5 \
#                         SCNN-Taiwanese-stream-test.h5 \
#                         SCNN-Chinese-CD-test.h5 \
#                         SCNN-Classical-test.h5 \
#                         SCNN-MIR-1k-train.h5 \
#                         SCNN-Instrumental-non-vocal.h5 \
#                         SCNN-A-Cappella-vocal.h5 \
#                         SCNN-test-hard.h5 \
#                         SCNN-RWC.h5 \
#         --classes 2 \
#         --sample_size 32000 1 \
#         --epochs 160 \
#         --batch_size 150 \
#         --loss categorical_crossentropy \
#         --optimizer adadelta \
#         --metrics accuracy \
#         --lr 1.0 \
#         --times 10 \
#         --num_gpus 3 \
#         --tag sync/Jamendo/20210123-12_SCNN18_SCNN-Jamendo-train_h5 \
#         --explainable \
#         --filter_x 40 \
#         --filter_y 80 \
#         --magnification 2

# for x in 30
# do
#     for y in 40
#     do
# for m in 0 2 3 4 5
# do
# ./run_tf2.sh python SCNN_train_with_less_sample.py \
#     --model SCNN18 \
#     --train_ds_path SCNN-Jamendo-train.h5 \
#     --val_ds_path SCNN-Jamendo-test.h5 \
#     --test_ds_paths SCNN-Jamendo-test.h5 \
#     --classes 2 \
#     --sample_size 32000 1 \
#     --epochs 160 \
#     --batch_size 150 \
#     --loss categorical_crossentropy \
#     --optimizer adadelta \
#     --metrics accuracy \
#     --lr 1.0 \
#     --times 10 \
#     --num_gpus 3 \
#     --seed 2 \
#     --train_ds_size 200
# done
#     done
# done


## enabled_transfer_learning

# ./run_tf2.sh python eval_voting.py \
#         --model SCNN18 \
#         --voting_tag sync/Jamendo/20210123-12_SCNN18_SCNN-Jamendo-train_h5 \
#         --voting_times 9 \
#         --train_ds_path SCNN-Jamendo-train.h5 \
#         --val_ds_path SCNN-Jamendo-test.h5 \
#         --test_ds_paths SCNN-RWC.h5 \
#         --test_add_retrain_sizes 100 200 \
#         --test_retrain_times 10 \
#         --classes 2 \
#         --sample_size 32000 1 \
#         --epochs 160 \
#         --batch_size 150 \
#         --num_gpus 3 \
#         --loss categorical_crossentropy \
#         --optimizer adadelta \
#         --metrics accuracy \
#         --lr 1.0 \
#         --verbose 0 \
#         --test_retrain_has_random
        # --enabled_transfer_learning \
        # --skip_origin

# ./run_tf2.sh python eval_voting.py \
#         --model SCNN18 \
#         --voting_tag sync/Jamendo/20210123-12_SCNN18_SCNN-Jamendo-train_h5 \
#         --voting_times 17 \
#         --train_ds_path SCNN-Jamendo-train.h5 \
#         --val_ds_path SCNN-Jamendo-test.h5 \
#         --test_ds_paths SCNN-Taiwanese-stream-train.h5 \
#                         SCNN-Classical-train.h5 \
#         --test_add_retrain_sizes 100 200 300 400 500 600 700  800 \
#         --test_retrain_times 10 \
#         --classes 2 \
#         --sample_size 32000 1 \
#         --epochs 160 \
#         --batch_size 150 \
#         --num_gpus 3 \
#         --loss categorical_crossentropy \
#         --optimizer adadelta \
#         --metrics accuracy \
#         --lr 1.0 \
#         --verbose 0 \
#         --test_retrain_has_random

# --use_saved_inital_weight

# ./run_tf1.sh python dl_keras/SCNN_18layer.py

# ./run_tf2.sh python train.py \
#     --model SCNN18_MaxPool12 \
#     --train_ds_path SCNN-Jamendo-train.h5 \
#     --val_ds_path SCNN-Jamendo-test.h5 \
#     --test_ds_paths SCNN-Jamendo-test.h5 \
#                     SCNN-FMA-C-1-fixed-test.h5 \
#                     SCNN-FMA-C-2-fixed-test.h5 \
#                     SCNN-KTV-test.h5 \
#                     SCNN-MIR-1k-train.h5 \
#                     SCNN-Instrumental-non-vocal.h5 \
#                     SCNN-A-Cappella-vocal.h5 \
#                     SCNN-Taiwanese-stream-test.h5 \
#                     SCNN-Taiwanese-CD-test.h5 \
#                     SCNN-Chinese-CD-test.h5 \
#                     SCNN-Classical-test.h5 \
#                     SCNN-test-hard.h5 \
#                     SCNN-RWC.h5 \
#                     SCNN-Classical-train.h5 \
#                     SCNN-FMA-C-1-fixed-train.h5 \
#                     SCNN-FMA-C-2-fixed-train.h5 \
#                     SCNN-KTV-train.h5 \
#                     SCNN-Taiwanese-CD-train.h5 \
#                     SCNN-Taiwanese-stream-train.h5 \
#                     SCNN-Chinese-CD-train.h5 \
#     --classes 2 \
#     --sample_size 32000 1 \
#     --epochs 160 \
#     --batch_size 150 \
#     --loss categorical_crossentropy \
#     --optimizer adadelta \
#     --metrics accuracy \
#     --lr 1.0 \
#     --times 21 \
#     --num_gpus 3 \
#     --verbose 1 \
#     --training

# docker kill tensorboard
# for m in 0
# do
python train.py \
    --model SCNN18 \
    --train_ds_path SCNN-FMA-C-1-fixed-train.h55 \
    --val_ds_path SCNN-FMA-C-1-fixed-test.h5 \
    --test_ds_paths SCNN-FMA-C-1-fixed-train.h5 \
                    SCNN-Jamendo-test.h5 \
                    SCNN-FMA-C-1-fixed-test.h5 \
                    SCNN-FMA-C-2-fixed-test.h5 \
                    SCNN-KTV-test.h5 \
                    SCNN-Taiwanese-CD-test.h5 \
                    SCNN-Taiwanese-stream-test.h5 \
                    SCNN-Chinese-CD-test.h5 \
                    SCNN-Classical-test.h5 \
                    SCNN-MIR-1k-train.h5 \
                    SCNN-Instrumental-non-vocal.h5 \
                    SCNN-A-Cappella-vocal.h5 \
                    SCNN-test-hard.h5 \
                    SCNN-RWC.h5 \
    --classes 2 \
    --sample_size 32000 1 \
    --epochs 160 \
    --batch_size 150 \
    --loss categorical_crossentropy \
    --optimizer adadelta \
    --metrics accuracy \
    --lr 1.0 \
    --times 21 \
    --num_gpus 3 \
    --tag FMA-C-1/2021-02-13_11_SCNN18_SCNN-FMA-C-1-fixed-train_h5_2GPU
# done