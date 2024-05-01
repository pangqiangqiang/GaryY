# 普通版

log_path='./log'
mkdir -p $log_path
DATE=$(date +%Y-%m-%d-%H_%M_%S)
python -u ./cnn_main.py | tee $log_path/training_$DATE.log


