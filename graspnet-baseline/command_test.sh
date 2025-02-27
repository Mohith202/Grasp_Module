# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs_v2/dump_rs --checkpoint_path logs_v2/log_rs/checkpoint.tar --camera realsense --dataset_root /ssd_scratch/mohit.g/
CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_rs --checkpoint_path logs/log_rs/checkpoint.tar --camera realsense --dataset_root /ssd_scratch/mohit.g/
# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_kn --checkpoint_path logs/log_kn/checkpoint.tar --camera kinect --dataset_root /data/Benchmark/graspnet
