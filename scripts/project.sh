cd ..
python main.py --query_method project1 --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 0 --dataset ood_mnist --mode mln_sn
python main.py --query_method project2 --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 0 --dataset ood_mnist --mode mln_sn
python main.py --query_method project1 --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 0 --dataset ood_mnist --mode mln_sn
python main.py --query_method project2 --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 0 --dataset ood_mnist --mode mln_sn

python main.py --query_method project1 --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 0 --mode mln_sn
python main.py --query_method project2 --init_dataset 50 --query_step 20 --query_size 20 --epoch 100 --id 2 --gpu 0 --mode mln_sn
python main.py --query_method project1 --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 0 --mode mln_sn
python main.py --query_method project2 --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 0 --mode mln_sn
