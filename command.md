# docker创建命令

docker run -i -t -d -v /home/lvjiaqi:/code -p 7788:22 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 /bin/bash

docker run -i -t -d -v /home/lvjiaqi:/code -v /nvme/home/lvjiaqi/model:/code/model -p 8899:22 lvjiaqi:llm /bin/bash

docker run --entrypoint /bin/bash 906ef1b523b3

# 挂载
mount -o remount,rw /nvme/home/lvjiaqi/model /code/model

# docker 启动命令

docker exec  --privileged -u root -it 906ef1b523b3 /bin/bash
docker exec -it 14652581dd2b /bin/bash


# docker 重启
docker restart -v /home/lvjiaqi:/code -v /nvme/home/lvjiaqi/model:/code/model -it 618569423f85
docker restart af3745a8fbeb
docker start af3745a8fbeb

# tmux命令

tmux new -s <session-name>
tmux ls
tmux attach -t 0
tmux attach -t <session-name>
tmux kill-session -t 0
tmux kill-session -t <session-name>

# model path
../model/CodeLlama-7b-hf
./llama2-13b-orca-8k-3319

# 保存容器为docker镜像
docker commit 906ef1b523b3 lvjiaqi:llm
docker save -o lvjiaqi-llm.img lvjiaqi:llm

# 设置代理

proxy_on
proxy_off

export http_proxy=http://lvjiaqi:SmJ#Jz+9eq@10.1.8.50:33128/
export https_proxy=http://lvjiaqi:SmJ#Jz+9eq@10.1.8.50:33128/

export http_proxy=http://lvjiaqi:SmJ%23Jz+9eq@10.1.8.50:33128/
export https_proxy=http://lvjiaqi:SmJ%23Jz+9eq@10.1.8.50:33128/
export HTTP_PROXY=http://lvjiaqi:SmJ%23Jz+9eq@10.1.8.50:33128/
export HTTPS_PROXY=http://lvjiaqi:SmJ%23Jz+9eq@10.1.8.50:33128/

unset http_proxy
unset https_proxy

# 服务器IP

10.140.0.189

# anaconda配置

conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free/
conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/msys2/

conda config --show channels

# 软链接

ln -s /home/lvjiaqi /nvme/home/lvjiaqi/model
ln -s /nvme/home/lvjiaqi/model /home/lvjiaqi 
unlink /nvme/home/lvjiaqi


# jupyter notebook

argon2:$argon2id$v=19$m=10240,t=10,p=8$S5bPOGnnFGtMbqtkta3I0A$4X6hjoKhLGGkJE/BeWBZBhZ7Z+LVW49twORnAObmbZc


 echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDPC7IIYFjXxUQQVIChaOcZ0UfCnTtkQ3yTJmkh7Y0wYhgVYZhYo9SkKoBdAaKL7II8EnTN5tmTRNTZiwTIQU6bT/PPmcBXAkXBBQPWZ4N5JbLNnuxrs/neysEi/iqKpbOpOl+JJOVmpVZc78mDmUVVZtI+Yg+7F+gny8sfSMwyl6OSgG/UlaAq2mioBPPS2J9r1yhU/oyCPIk8kaPmiTvL9PNERzuWouGNchczx2ioTButW5dQuPz+J62GyExfiZx5lieH0gCS0cOOH/oSB0POeXhqjoPXqgP3qFN8qoXtFc2DOA88YQ4T7G4xCSj/Z1Xa26NY8OadOLMBR14l8IO9RyFZuGJkfaHicQCctZloiLoDN2Co+0aNFaS0YB7vSSQGSWtMnY+tMPHAqcbrTwbTBC6PvNwVbCwO09esMyAGdWWEIdJxrs5sndupBTTvag/qiasGd3E50fqtwq1nldAgnhi0TCw5pOaIgZIdh7qkbhGR6q5HniJQ4ZK/jOKHOUs= 1031903858@qq.com" >> ~/.ssh/authorized_keys

# gcc g++ 版本更新
https://blog.csdn.net/CharlieVV/article/details/111242143
## 查看可用版本
apt-cache policy g++-9
## 安装
apt-get install g++-9=9.5.0-1ubuntu1~22.04
## 查询本机gcc已安装的版本
ls /usr/bin/g++*
## 确定优先级
update-alternatives --install /usr/bin/gcc gcc /usr/bin/g++-9 40
update-alternatives --install /usr/bin/gcc gcc /usr/bin/g++-11 50
update-alternatives --config g++
