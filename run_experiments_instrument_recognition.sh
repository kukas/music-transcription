PARAMS="--spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --batchnorm 1 --context_width 0 --iterations 200000 --frame_width 1024 --batch_size 16"
LEARNING_PARAMS2="--stop_if_too_slow 60 --threads 6 --evaluate_small_every 5000 --evaluate_every 50000"

# baseline
python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture LY --filters 32 --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 1 --batchnorm 1 --iterations 200000 --evaluate

# HCNN-fc
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 32
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 68
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 16
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 34
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 8
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 17
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 36
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 77
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 28
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 59
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 24
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 50
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 20
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 42

# HCNN-avg
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 8 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 16 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 64 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 48 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 72 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 80 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 96 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0

python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 8  --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 16 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 24 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 40 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 48 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
