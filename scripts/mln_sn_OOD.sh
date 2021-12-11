cd ..
python main.py --query_method epistemic --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method aleatoric --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method pi_entropy --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method maxsoftmax --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method entropy --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method random --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method bald --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method density --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 2 --dataset ood_mnist --mode mln_sn
python plot_ood.py --id 2 --mode mln_sn

python main.py --query_method epistemic --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method aleatoric --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method pi_entropy --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method maxsoftmax --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method entropy --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method random --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method bald --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2 --dataset ood_mnist --mode mln_sn
python main.py --query_method density --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2 --dataset ood_mnist --mode mln_sn
python plot_ood.py --id 3 --mode mln_sn