# $1: GPU番号. デフォルトは0.
# $2: 実行番号. デフォルトは1.
# $3: tmuxのセッション名. デフォルトは空.指定しない場合はローカルで実行.

conda activate tta
cd /home/syamagami/lab/tta/test-time-adaptation/classification

COMMAND="CUDA_VISIBLE_DEVICES=${1:-0}  exec_num=${2:-1}  python  test_time.py  --cfg  cfgs/cifar10_c/rmt.yaml"
if [ -n "$3" ]; then
    # 第3引数が存在する場合の処理. tmux内で実行する. $3はtmuxのセッション名.
    tmux -2 new -d -s $3
    tmux send-key -t $3.0 "$COMMAND" ENTER
else
    # 第3引数が存在しない場合の処理. そのまま実行.
    eval $COMMAND
fi
