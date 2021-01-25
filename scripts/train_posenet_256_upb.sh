TRAIN_SET=/mnt/storage/workspace/andreim/nemodrive/upb_data/dataset/train+val_frames/
python train.py $TRAIN_SET \
--pretrained-disp=pretrained_models/NeurIPS_Models/depth/cs+k_depth.tar \
--pretrained-pose=pretrained_models/NeurIPS_Models/pose/cs+k_pose.tar \
--dispnet DispResNet \
--num-scales 1 \
-b16 -s0.1 -c0.5 --epoch-size 2000 --sequence-length 3 \
--with-mask \
--with-ssim \
--name cs+k_posenet_256_upb \
--learning-rate 1e-5
