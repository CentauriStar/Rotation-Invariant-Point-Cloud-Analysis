CUDA_VISIBLE_DEVICES=0 python cls_train.py --dataset "ModelNetNormal" \
--batch_size 32 --batch_size_test 64 \
--dir_name log/no_normal/modelnet40_rotz/best --test 1 --training 0 --test_final 1 \
--network "ModelNet" --sample_points 1024 \
--nepoch 20 --lrate 0.001 --k 20 \
--axes glob bary \
--rot_z \
--dp 0.8 \
--use_annl --use_sgd