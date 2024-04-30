CUDA_VISIBLE_DEVICES=0 python cls_train.py --dataset "ScanObjectNNCls" \
--batch_size 32 --batch_size_test 128 \
--dir_name log/scan_rotz/best --test 1 --training 0 --test_final 1 \
--network "ScanObjectNN" --sample_points 1024 \
--nepoch 20 --lrate 0.001 --k 20 \
--rot_z \
--dp 0.8 \
--use_annl --use_sgd --o3d_normal \
--with_norm