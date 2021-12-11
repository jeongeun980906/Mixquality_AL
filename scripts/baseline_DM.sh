cd ..
python main.py --mode base --query_method coreset --init_dataset 100 --query_step 20 --query_size 50 --epoch 100 --id 3 --gpu 1
python main.py --mode base --query_method maxsoftmax --init_dataset 100 --query_step 20 --query_size 50 --epoch 100 --id 3 --gpu 1
python main.py --mode base --query_method entropy --init_dataset 100 --query_step 20 --query_size 50 --epoch 100 --id 3 --gpu 1
python main.py --mode base --query_method random --init_dataset 100 --query_step 20 --query_size 50 --epoch 100 --id 3 --gpu 1
python plot_ood.py --id 3 --mode base