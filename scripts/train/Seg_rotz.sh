CUDA_VISIBLE_DEVICES=0 python seg_train.py --dataset "ShapeNetPart" --task "seg" \
--batch_size 24 --batch_size_test 64 \
--dir_name log/no_normal/seg_rotz --test 1 \
--sample_points 2048 \
--nepoch 400 --lrate 0.001 --k 40 \
--axes glob bary \
--rot_z \
--dp 0.8 \
--use_sgd --use_annl
