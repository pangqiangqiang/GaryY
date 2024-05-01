# 普通版

log_path='./log'
mkdir -p $log_path
DATE=$(date +%Y-%m-%d-%H_%M_%S)
python -u ./bert_main.py \
    --data_dir ./data/sst2_shuffled.tsv\
    --train_ratio 0.8\
    --class_num 2\
    --max_length 64\
    --bert_name_path ../model/google-bert/bert-base-uncased\
    --batch_size 80\
    --epochs 10\
    --max_iter 1000 --logging_step 20\
    --resume 0\
    --check_point_path ./save/ 2>&1 | tee $log_path/training_$DATE.log

# python 003/bert_bert_main.py \
#     --data_dir 003/data/sst2_shuffled.tsv\
#     --train_ratio 0.8\
#     --class_num 2\
#     --max_length 64\
#     --bert_name_path model/BAAI/bge-base-en-v1.5\
#     --batch_size 80\
#     --epochs 10\
#     --max_iter 1000 --logging_step 20\
#     --resume 0\
#     --check_point_path ./003/save/

# python 003/bert_main.py \
#     --data_dir 003/data/sst2_shuffled.tsv\
#     --train_ratio 0.8\
#     --class_num 2\
#     --max_length 64\
#     --bert_name_path model/BAAI/bge-large-en-v1.5\
#     --batch_size 80\
#     --epochs 10\
#     --max_iter 1000 --logging_step 20\
#     --resume 0\
#     --check_point_path ./003/save/

# python 003/bert_main.py \
#     --data_dir 003/data/sst2_shuffled.tsv\
#     --train_ratio 0.8\
#     --class_num 2\
#     --max_length 64\
#     --bert_name_path model/51la5/distilbert-base-sentiment\
#     --batch_size 80\
#     --epochs 10\
#     --max_iter 1000 --logging_step 20\
#     --resume 0\
#     --check_point_path ./003/save/




# 加入对抗训练版


# 加入指数平均移动版
# python 003/bert_main.py \
#     --data_dir 003/data/sst2_shuffled.tsv\
#     --train_ratio 0.8\
#     --class_num 2\
#     --max_length 64\
#     --bert_name_path model/google-bert/bert-base-uncased\
#     --batch_size 80\
#     --epochs 10\
#     --max_iter 1000 --logging_step 20\
#     --resume 0\
#     --check_point_path ./003/save/\
#     --use_ema 1\

# 加入噪声微调版



