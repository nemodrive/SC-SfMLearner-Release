TRAIN_SET=/raid/workspace/andreim/nemodrive/upb_data/dataset
python train.py $TRAIN_SET \
--pretrained-disp= \
--pretrained-pose= \
--dispnet DispResNet \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 2000 --sequence-length 3 \
--with-mask \
--with-ssim \
--name cs+k_posenet_256_upb
