CUDA_VISIBLE_DEVICES=4,5,6,7, python -m torch.distributed.launch --nproc_per_node=4 main.py --tag resnet50-sgd-lr-3 --cfg config/resnet/cifar10-resnet-train-sgd.yaml --dataset cifar10 --data_path /data1/hzh/cifar10 --batch_size 128 --output output