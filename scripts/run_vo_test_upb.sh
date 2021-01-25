DATASET_DIR=/mnt/storage/workspace/andreim/nemodrive/upb_data/dataset/test_frames/
OUTPUT_DIR=results/vo/cs+k_pose_upb/

POSE_NET=checkpoints/cs+k_posenet_256_upb/01-23-18\:55/exp_pose_checkpoint.pth.tar

# save the visual odometry results to "results_dir/09.txt"
python test_vo_upb.py \
--sequence 001fd5e96d7134f509 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

# show the trajectory with gt. note that use "-s" for global scale alignment
evo_traj kitti -s $OUTPUT_DIR/01fd5e96d7134f50.txt -p --plot_mode=xz

