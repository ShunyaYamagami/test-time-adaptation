# $1: GPU番号. デフォルトは0.
# $2: 実行番号. デフォルトは1.
# $tmux_session: tmuxのセッション名. デフォルトは空.指定しない場合はローカルで実行.
gpu_i=${1:-0}
exec_num=${2:-1}
tmux_session=$3

conda activate tta

###### 実行. 
# 実行コマンド
COMMAND="CUDA_VISIBLE_DEVICES=$gpu_i  exec_num=$exec_num  python  test_time.py  --cfg  cfgs/cifar10_c/rmt.yaml"

if [ -n "$tmux_session" ]; then
    # 第3引数が存在する場合の処理. tmux内で実行する. $tmux_sessionはtmuxのセッション名.
    tmux -2 new -d -s $tmux_session
    tmux send-key -t $tmux_session.0 "$COMMAND" ENTER
else
    # 第3引数が存在しない場合の処理. そのまま実行.
    eval $COMMAND
fi


# CUDA_VISIBLE_DEVICES=1  exec_num=1  python  test_time.py  --cfg  cfgs/cifar10_c/rmt.yaml
# CUDA_VISIBLE_DEVICES=1  exec_num=1  python  pretrain.py  --cfg  cfgs/cifar10_c/rmt.yaml
# . cmd.sh  0  1  0
# . cmd.sh  1  1  1
# . cmd.sh  2  1  2
# . cmd.sh  3  1  3