CUDA_VISIBLE_DEVICES=0 python cls_train.py --dataset "ScanObjectNNCls" \
--batch_size 32 --batch_size_test 128 \
--dir_name log/scan_so3 --test 1 \
--network "ScanObjectNN" --sample_points 1024 \
--nepoch 500 --lrate 0.001 --k 20 \
--rand_rot \
--dp 0.8 \
--use_sgd --use_annl --o3d_normal \
--with_norm