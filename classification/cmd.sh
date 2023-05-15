conda activate tta
cd /home/syamagami/lab/tta/test-time-adaptation/classification
CUDA_VISIBLE_DEVICES=1  python test_time.py --cfg cfgs/cifar10_c/rmt.yaml