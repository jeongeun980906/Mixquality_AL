cd ..
python main.py --mode bald --query_method bald --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2
python main.py --mode bald --query_method mean_std --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2
python main.py --mode bald --query_method maxsoftmax --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2
python main.py --mode bald --query_method entropy --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2
python main.py --mode bald --query_method random --init_dataset 100 --query_step 10 --query_size 50 --epoch 100 --id 3 --gpu 2
python plot_dirty.py --mode bald --id 3 --gpu 2
