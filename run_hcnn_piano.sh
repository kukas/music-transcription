set -xe

PARAMS="--spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8"
LEARNING_PARAMS="--stop_if_too_slow 60 --threads 6 --evaluate_small_every 5000 --evaluate_every 20000 --iterations 100000 --learning_rate 0.001 --learning_rate_decay 0.8 --learning_rate_decay_steps 20000 --batch_size 8 --evaluate"

python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 1 --overtone_stacking 4 --conv_ctx 3 --dilations -2 -1 --use_bias 0 --batchnorm 1 --annotations_per_window 5 --context_width 17920
