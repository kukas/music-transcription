PARAMS="--spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8"
LEARNING_PARAMS="--stop_if_too_slow 60 --threads 6 --evaluate_small_every 5000 --evaluate_every 20000 --iterations 100000 --learning_rate 0.001 --learning_rate_decay 0.8 --learning_rate_decay_steps 20000 --batch_size 8 --evaluate"

# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 4 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 4 --undertone_stacking 2 --overtone_stacking 3 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 4 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 4 --undertone_stacking 4 --overtone_stacking 5 --conv_ctx 1

# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 8 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 8 --undertone_stacking 2 --overtone_stacking 3 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 8 --undertone_stacking 4 --overtone_stacking 5 --conv_ctx 1

# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 12 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 12 --undertone_stacking 2 --overtone_stacking 3 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 12 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 12 --undertone_stacking 4 --overtone_stacking 5 --conv_ctx 1

# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 4 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 32 --stacks 4 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 64 --stacks 4 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1

# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 32 --stacks 8 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 64 --stacks 8 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1

python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 32 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 3 1

python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 12 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 32 --stacks 12 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 64 --stacks 12 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 8 --stacks 16 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 16 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 32 --stacks 16 --undertone_stacking 1 --overtone_stacking 2 --conv_ctx 1


python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 4 --undertone_stacking 2 --overtone_stacking 3 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 24 --stacks 4 --undertone_stacking 2 --overtone_stacking 3 --conv_ctx 1

python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 2 --overtone_stacking 3 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 24 --stacks 8 --undertone_stacking 2 --overtone_stacking 3 --conv_ctx 1

python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 12 --undertone_stacking 2 --overtone_stacking 3 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 24 --stacks 12 --undertone_stacking 2 --overtone_stacking 3 --conv_ctx 1


python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 4 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 24 --stacks 4 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 1

python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 24 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 1

python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 12 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 1
python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 24 --stacks 12 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 1


python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 4 --undertone_stacking 4 --overtone_stacking 5 --conv_ctx 1

python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 4 --overtone_stacking 5 --conv_ctx 1

python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 12 --undertone_stacking 4 --overtone_stacking 5 --conv_ctx 1
