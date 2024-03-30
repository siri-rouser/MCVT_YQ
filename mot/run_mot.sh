# #!/bin/bash
# cd tool
# python pre_process.py

# # cd ..
# if [ ! -d "./build" ];then # -d check is here a directory called ./build
#     mkdir build
# else
#     rm -rf ./build/* # if here is ./build dir, remove everything inside
# fi
# cd build && cmake .. && make -j4 # if command 1 succuess, command 2 will run and then command 3; && is a AND in here
# ./city_tracker

cd tool
python post_precess.py

#seqs=(c041 c042 c043 c044 c045 c046)
# seqs=(c042)

# cd tool
# TrackOneSeq(){
#     seq=$1
#     config=$2
#     echo save_sot $seq with ${config}
#     python save_mot.py ${seq} pp ${config}
# }

# for seq in ${seqs[@]}
# do 
#     TrackOneSeq ${seq} $1 &
# done
# wait