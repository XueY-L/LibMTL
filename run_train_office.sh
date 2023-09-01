python3 train_office.py \
    --seed 42 \
    --gpu_id 3  \
    --multi_input  \
    --weighting MGDA \
    --save_path './results' \
    --scheduler step \
    --mgda_gn l2 \
    --arch HPS \
    # --rep_grad
