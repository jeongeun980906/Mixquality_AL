cd ..
# python main.py --mode ddu --query_method epistemic --init_dataset 50 --query_step 20 --query_size 20 --epoch 50 --id 2 --gpu 3 --dataset ood_mnist
# python main.py --mode ddu --query_method aleatoric --init_dataset 50 --query_step 20 --query_size 20 --epoch 50 --id 2 --gpu 3 --dataset ood_mnist
# python main.py --mode ddu --query_method random --init_dataset 50 --query_step 20 --query_size 20 --epoch 50 --id 2 --gpu 3 --dataset ood_mnist
# python plot_ood.py --id 2 --mode ddu --dataset ood_mnist

python main.py --mode ddu --query_method epistemic --init_dataset 100 --query_step 10 --query_size 50 --epoch 50 --id 3 --gpu 2 --dataset ood_mnist
python main.py --mode ddu --query_method aleatoric --init_dataset 100 --query_step 10 --query_size 50 --epoch 50 --id 3 --gpu 2 --dataset ood_mnist
python main.py --mode ddu --query_method random --init_dataset 100 --query_step 10 --query_size 50 --epoch 50 --id 3 --gpu 2 --dataset ood_mnist
python plot_ood.py --id 3 --mode ddu --dataset ood_mnist