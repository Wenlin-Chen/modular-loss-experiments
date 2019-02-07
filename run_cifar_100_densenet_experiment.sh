python src/run_configuration.py --n-trials 3 --dataset CIFAR-100 --augment-data --network-type densenet --densenet-k 12 --densenet-depth 100 --densenet-reduction 0.5 --densenet-bottleneck --batch-size 64 --epochs 300 --momentum 0.9 --learning-rates 0.1 --learning-rate-decay-milestones 150 225 --learning-rate-decay-factor 0.1 --weight-decay 0.0001 --n-modules 4 --output-directory densenet-100-12-4 --lambda-values 0.0 0.5 0.9 1.0 --seed 99283

python src/run_configuration.py --n-trials 3 --dataset CIFAR-100 --augment-data --network-type densenet --densenet-k 8 --densenet-depth 82 --densenet-reduction 0.5 --densenet-bottleneck --batch-size 64 --epochs 300 --momentum 0.9 --learning-rates 0.1 --learning-rate-decay-milestones 150 225 --learning-rate-decay-factor 0.1 --weight-decay 0.0001 --n-modules 8 --output-directory densenet-82-8-8 --lambda-values 0.0 0.5 0.9 1.0 --seed 99283

python src/run_configuration.py --n-trials 3 --dataset CIFAR-100 --augment-data --network-type densenet --densenet-k 6 --densenet-depth 64 --densenet-reduction 0.5 --densenet-bottleneck --batch-size 64 --epochs 300 --momentum 0.9 --learning-rates 0.1 --learning-rate-decay-milestones 150 225 --learning-rate-decay-factor 0.1 --weight-decay 0.0001 --n-modules 16 --output-directory densenet-64-6-16 --lambda-values 0.0 0.5 0.9 1.0 --seed 99283