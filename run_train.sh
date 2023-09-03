python3 train_domainnet_fusion.py \
    --seed 42 \
    --gpu_id 0  \
    --bs 32 \
    --epochs 40 \
    --multi_input  \
    --weighting MGDA \
    --save_path './results' \
    --mgda_gn l2 \
    --arch HPS \
    --optim adam \
    --lr 1e-4 \
    # --scheduler None \
    # --rep_grad
