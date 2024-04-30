CUDA_VISIBLE_DEVICES=1 python cls_train.py --dataset "ModelNetNormal" \
--batch_size 32 --batch_size_test 64 \
--dir_name log/no_normal/modelnet40_so3 --test 1 --training 1 \
--network "ModelNet" --sample_points 1024 \
--nepoch 300 --lrate 0.001 --k 20 \
--axes glob bary \
--rand_rot \
--dp 0.8 \
--use_sgd --use_annl

