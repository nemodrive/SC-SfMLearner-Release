DATASET_DIR=/mnt/storage/workspace/andreim/nemodrive/upb_data/dataset/train+val_frames/
OUTPUT_DIR=results/vo/cs+k_pose_upb/

POSE_NET=checkpoints/exp_pose_model_best.pth.tar

# save the visual odometry results to "results_dir/09.txt"
python test_vo_upb.py \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR
