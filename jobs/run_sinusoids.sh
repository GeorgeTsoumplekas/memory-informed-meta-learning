# Base experiment for INPs on sinusoids
# ======================================
# # Train NP without knowledge
# python config.py  --project-name INPs_sinusoids --dataset set-trending-sinusoids --input-dim 1 --output-dim 1 --run-name-prefix np --use-knowledge False --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 400 --text-encoder set --knowledge-merge sum --seed 1
# python models/train.py

# # Train INP with knowledge as one or two parameters a, b, c
# python inp_config.py  --project-name INPs_sinusoids --dataset set-trending-sinusoids  --input-dim 1 --output-dim 1 --run-name-prefix inp_abc2 --use-knowledge True --noise 0.2 --min-num-context 0 --max-num-context 10 --num-targets 100 --batch-size 64 --num-epochs 400 --text-encoder set --knowledge-merge sum --knowledge-type abc2 --test-num-z-samples 32 --seed 2
# python models/train_inp.py

# Train Memory INP with knowledge as one or two parameters a, b, c
python memory_inp_config.py --project-name INPs_sinusoids \
                             --seed 2 \
                             --batch-size 64 \
                             --lr 1e-3 \
                             --lr-step-size 401 \
                             --lr-decay 0.1 \
                             --num-epochs 400 \
                             --input-dim 1 \
                             --output-dim 1 \
                             --run-name-prefix memory_inp_abc2 \
                             --dataset set-trending-sinusoids \
                             --min-num-context 0 \
                             --max-num-context 10 \
                             --num-targets 100 \
                             --noise 0.2 \
                             --knowledge-type abc2 \
                             --dataset-encoder-type self_attention \
                             --dataset-representation-dim 128 \
                             --set-transformer-num-heads 4 \
                             --set-transformer-num-inds 8 \
                             --set-transformer-ln True \
                             --set-transformer-hidden-dim 128 \
                             --set-transformer-num-seeds 1 \
                             --x-transf-dim 128 \
                             --xy-transf-dim 128 \
                             --xy-encoder-hidden-dim 128 \
                             --xy-encoder-num-hidden 2 \
                             --knowledge-representation-dim 128 \
                             --text-encoder set \
                             --set-embedding-num-hidden 2 \
                             --knowledge-encoder-num-hidden 2 \
                             --knowledge-dropout 0.3 \
                             --use-knowledge True \
                             --understanding-representation-dim 128 \
                             --knowledge-dataset-merge sum \
                             --knowledge-dataset-merger-hidden-dim 128 \
                             --knowledge-dataset-merger-num-hidden 1 \
                             --understanding-encoder-num-hidden 2 \
                             --data-interaction-mlp-num-hidden 2 \
                             --data-interaction-self-attention-hidden-dim 128 \
                             --data-interaction-self-attention-num-heads 4 \
                             --data-interaction-cross-attention-hidden-dim 128 \
                             --data-interaction-cross-attention-num-heads 4 \
                             --data-interaction-dim 128 \
                             --use-memory True \
                             --memory-slots 64 \
                             --memory-learning-rate 1 \
                             --memory-decay-rate 0.3 \
                             --memory-write-temperature 0.1 \
                             --data-interaction-understanding-merge sum \
                             --data-interaction-understanding-merger-hidden-dim 128 \
                             --data-interaction-understanding-merger-num-hidden 1 \
                             --latent-encoder-hidden-dim 128 \
                             --latent-encoder-num-hidden 1 \
                             --decoder-activation gelu \
                             --decoder-hidden-dim 128 \
                             --decoder-num-hidden 3 \
                             --test-num-z-samples 32 \
                             --train-num-z-samples 1 \
                             --run-name-prefix memory_inp_abc2
python models/train_memory_inp.py
