python3 train_domainnet.py \
    --seed 42 \
    --gpu_id 0  \
    --bs 32 \
    --epochs 40 \
    --multi_input  \
    --weighting MGDA \
    --save_path './results' \
    --scheduler step \
    --mgda_gn l2 \
    --arch HPS \
    --optim adam \
    --lr 1e-4 \

    # --rep_grad
