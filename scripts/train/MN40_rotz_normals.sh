CUDA_VISIBLE_DEVICES=0 python cls_train.py --dataset "ModelNetNormal" \
--batch_size 32 --batch_size_test 64 \
--dir_name log/modelnet40_rotz --test 1 \
--network "ModelNet" --sample_points 1024 \
--nepoch 300 --lrate 0.001 --k 20 \
--rot_z \
--dp 0.8 \
--use_sgd --use_annl \
--with_norm
