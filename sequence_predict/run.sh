nohup python ./src/main.py -g 5 -p 30 -b huabei --colsample_bytree 0.8 --eta 0.1 \
--max_depth 9 --min_child_weight 3 --n_estimates 200 --subsample 0.9 --gamma 0.1 1>./temp_file/huabei30.log &
nohup python ./src/main.py -g 5 -p 60 -b huabei --colsample_bytree 0.7 --eta 0.1 \
--max_depth 9 --min_child_weight 7 --n_estimates 200 --subsample 0.8 --gamma 0.1 1>./temp_file/huabei60.log &
nohup python ./src/main.py -g 5 -p 5 -b jiebei --colsample_bytree 0.8 --eta 0.07 \
--max_depth 5 --min_child_weight 3 --n_estimates 200 --subsample 0.7 1>./temp_file/jiebei5.log &
nohup python ./src/main.py -g 5 -p 30 -b jiebei --colsample_bytree 0.6 --eta 0.05 \
--max_depth 9 --min_child_weight 7 --n_estimates 200 --subsample 0.9 --gamma 0.1 1>./temp_file/jiebei30.log &
nohup python ./src/main.py -g 5 -p 60 -b jiebei --colsample_bytree 0.6 --eta 0.05 \
--max_depth 9 --min_child_weight 5 --n_estimates 200 --subsample 0.8 --gamma 0.1 1>./temp_file/jiebei60.log &
