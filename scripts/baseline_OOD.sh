cd ..
# python main.py --mode base --query_method coreset --init_dataset 100 --query_step 10 --query_size 50 --epoch 50 --id 2 --gpu 0 --dataset ood_mnist
# python main.py --mode base --query_method maxsoftmax --init_dataset 100 --query_step 10 --query_size 50 --epoch 50 --id 2 --gpu 0 --dataset ood_mnist
# python main.py --mode base --query_method entropy --init_dataset 100 --query_step 10 --query_size 50 --epoch 50 --id 2 --gpu 0 --dataset ood_mnist
# python main.py --mode base --query_method random --init_dataset 100 --query_step 10 --query_size 50 --epoch 50 --id 2 --gpu 0 --dataset ood_mnist
# python plot_ood.py --id 2 --mode base

python main.py --mode base --query_method coreset --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 0 --dataset ood_mnist
python main.py --mode base --query_method maxsoftmax --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 0 --dataset ood_mnist
python main.py --mode base --query_method entropy --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 0 --dataset ood_mnist
python main.py --mode base --query_method random --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 0 --dataset ood_mnist
python plot_ood.py --id 3 --mode base