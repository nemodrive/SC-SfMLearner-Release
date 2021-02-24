DATASET_DIR=/mnt/storage/workspace/andreim/kitti/data_odometry_color/dataset/sequences/
OUTPUT_DIR=results/vo/cs+k_pose_kitti/

POSE_NET=checkpoints/exp_pose_model_best.pth.tar

# save the visual odometry results to "results_dir/09.txt"
python test_vo.py \
--img-height 256 --img-width 832 \
--sequence 01 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

# show the trajectory with gt. note that use "-s" for global scale alignment
# evo_traj kitti -s $OUTPUT_DIR/09.txt --ref=./kitti_eval/09.txt -p --plot_mode=xz

