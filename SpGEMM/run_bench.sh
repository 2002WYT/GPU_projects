python3 ./scripts/run_suite.py \
    --matrix-dir /home/opsadmin/wangyitong/mtxs/ \
    --bin-dir build \
    --results-dir results \
    --b-mode "${B_MODE:-transpose}" \
    --warmup 2 --repeat 5 --timeout 900
