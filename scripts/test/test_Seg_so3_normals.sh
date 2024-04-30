CUDA_VISIBLE_DEVICES=0 python seg_train.py --dataset "ShapeNetPart" --task "seg" \
--batch_size 23 --batch_size_test 64 \
--dir_name log/seg_so3/best --test 1 --training 0 --test_final 1 \
--sample_points 2048 \
--nepoch 20 --lrate 0.001 --k 40 \
--rand_rot \
--dp 0.8 \
--use_sgd --use_annl \
--with_norm