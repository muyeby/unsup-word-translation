dev=3
src=en
tgt=fi
norm='renorm,center,renorm'
DATA_DIR=/disk/xfbai/corpus/monolingual

random_list=$(python3 -c "import random; random.seed(0); print(' '.join([str(random.randint(0, 1000)) for _ in range(10)]))") # random seeds
#random_list=(265)
echo $random_list
for s in ${random_list[@]}
do
echo $s
CUDA_VISIBLE_DEVICES=$dev python unsupervised.py  \
	--src_lang $src \
	--tgt_lang $tgt \
	--src_emb $DATA_DIR/$src.emb.txt \
	--tgt_emb $DATA_DIR/$tgt.emb.txt \
	--exp_path tune-wacky-new \
	--exp_name $src-$tgt/$s \
	--exp_id $norm \
	--seed $s \
	--adversarial True \
	--n_epochs 5 \
	--map_beta 0.01 \
	--epoch_size 1000000 \
	--max_vocab_A 200000 \
	--max_vocab_B 200000 \
	--dis_most_frequent_AB 50000 \
	--dis_most_frequent_BA 50000 \
   	--normalize_embeddings $norm \
	--emb_dim_autoenc 350 \
	--dis_lambda 1 \
	--cycle_lambda 5 \
	--reconstruction_lambda 1 \
	--dico_eval vecmap
done
