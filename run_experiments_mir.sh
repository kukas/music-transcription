set -xe

PARAMS="--spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8"
LEARNING_PARAMS="--stop_if_too_slow 60 --threads 6 --evaluate_small_every 5000 --evaluate_every 20000 --iterations 100000 --learning_rate 0.001 --learning_rate_decay 0.8 --learning_rate_decay_steps 20000 --batch_size 8 --evaluate"
LEARNING_PARAMS2="--stop_if_too_slow 60 --threads 6 --evaluate_small_every 5000 --evaluate_every 50000"


# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture baseline --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture JY --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture JY --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 0 --batchnorm 1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture JY --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 0 --batchnorm 1 --logdir models/0312_201033-mir-sYunNingHung_cqt-sus0-sos0-aJY-b1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8 --architecture baseline


# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture JY --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 0 --batchnorm 1 --evaluate --logdir models/0313_130345-mir-sYunNingHung_cqt-sus0-sos0-aJY-b1 --save_salience
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture baseline --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 0 --batchnorm 1 --evaluate --logdir models/0313_153413-mir-sYunNingHung_cqt-sus0-sos0-abaseline-b1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 2 --batchnorm 1 --context_width 0 --undertone_stacking 0 --overtone_stacking 0 --last_conv_kernel 1 72 --evaluate --logdir models/0313_163217-mir-cw0-scqt-sus0-sos2-adeep_hcnn-us0-os0-lck1,72-b1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 0 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --evaluate --logdir models/0313_165210-mir-cw0-scqt-sus8-sos9-adeep_hcnn-us2-os3-lck1,72-b1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 1024 --conv_ctx 3 3 1 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --evaluate --logdir models/0313_173144-mir-cw1024-scqt-sus8-sos9-adeep_hcnn-us2-os3-cc3,3,1-lck1,72-b1/

# bigger training
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture JY --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 0 --batchnorm 1 --evaluate --logdir models/0313_191903-mir-!withoutvalid!-sYunNingHung_cqt-sus0-sos0-aJY-b1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture JY --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 0 --batchnorm 1 --annotations_per_window 250 --batch_size 10

# JY hcnn
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture JY --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --iterations 300000
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 1024 --conv_ctx 3 3 1 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024 --evaluate --logdir models/0313_234217-mir-fw1024-cw1024-scqt-sus8-sos9-adeep_hcnn-us2-os3-cc3,3,1-lck1,72-lpavg-b1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 1024 --conv_ctx 3 3 1 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling max --iterations 200000 --frame_width 1024 --evaluate --logdir models/0314_005843-mir-fw1024-cw1024-scqt-sus8-sos9-adeep_hcnn-us2-os3-cc3,3,1-lck1,72-lpmax-b1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 1024 --conv_ctx 3 3 1 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 6 --last_pooling maxoct --iterations 200000 --frame_width 1024 --evaluate --logdir models/0314_020608-mir-fw1024-cw1024-scqt-sus8-sos9-adeep_hcnn-us2-os3-cc3,3,1-lck1,6-lpmaxoct-b1

# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 1024 --conv_ctx 3 3 1 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024 --class_weighting 0 --evaluate --logdir models/0314_101514-mir-fw1024-cw1024-scqt-sus8-sos9-adeep_hcnn-cw0-us2-os3-cc3,3,1-lck1,72-lpavg-b1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 2048 --conv_ctx 3 3 3 1   --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024
#python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 4096 --conv_ctx 3 3 3 3 1 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024

#python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 2048 --conv_ctx 3 3 1  --filters 16 --stacks 4 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024
#python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 2048 --conv_ctx 3 3 1  --filters 16 --stacks 8 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024

# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 2048 --conv_ctx 3 3 1  --filters 32 --stacks 4 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 2048 --conv_ctx 3 3 1  --filters 32 --stacks 8 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 2048 --conv_ctx 3 3 1  --filters 16 --stacks 12 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 2048 --conv_ctx 3 3 1  --filters 16 --stacks 16 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 2048 --conv_ctx 3 3 1  --filters 16 --stacks 4 --undertone_stacking 3 --overtone_stacking 4 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 9 --batchnorm 1 --context_width 2048 --conv_ctx 3 3 1  --filters 16 --stacks 4 --undertone_stacking 4 --overtone_stacking 5 --last_conv_kernel 1 72 --last_pooling avg --iterations 200000 --frame_width 1024
# exit()

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

# slow hcnn / fast hcnn test
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 32 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 3 1

# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 32 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 3 1 --faster_hcnn 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 32 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 3 1 --faster_hcnn 0
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 32 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 --faster_hcnn 0
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 --faster_hcnn 0
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 --faster_hcnn 0 --use_bias 0
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 --faster_hcnn 1 --use_bias 0
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 --faster_hcnn 0 --use_bias 0
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 --faster_hcnn 0 --use_bias 0 --batchnorm 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 3 --overtone_stacking 4 --conv_ctx 3 --faster_hcnn 1 --use_bias 0 --batchnorm 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 1 --overtone_stacking 4 --conv_ctx 3 --use_bias 0 --batchnorm 1
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 1 --overtone_stacking 4 --conv_ctx 3 --use_bias 0 --batchnorm 1 --batch_size 16
# python -u spectrogram_mf0.py $PARAMS $LEARNING_PARAMS  --filters 16 --stacks 8 --undertone_stacking 1 --overtone_stacking 4 --conv_ctx 3 --dilations -2 -1 --use_bias 0 --batchnorm 1 --batch_size 8 --annotations_per_window 5 --context_width 8960


PARAMS="--spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --batchnorm 1 --context_width 2048 --iterations 200000 --frame_width 1024"
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 32 --stacks 4 --undertone_stacking 2 --overtone_stacking 3 --last_conv_kernel 1 72 --last_pooling avg
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 4 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 4 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 2 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 8 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 8 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 2 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 12 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 12 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 2 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 16 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 16 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 2 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 20 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 20 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 2 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 24 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 24 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 2 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 28 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 28 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 2 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 32 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 32 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 2 --overtone_stacking 3


# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 4 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 8 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 12 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 16 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 20 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 24 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 28 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 32 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate

# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 16 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 20 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 24 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 28 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 32 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 64 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1  --filters 128 --stacks 4 --last_conv_kernel 1 72 --last_pooling avg --undertone_stacking 0 --overtone_stacking 1 --evaluate




# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1 --filters 8 --stacks 4 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1 --filters 16 --stacks 4 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1 --filters 32 --stacks 4 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1 --filters 4 --stacks 4 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 5 --last_pooling globalavg --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1 --filters 8 --stacks 4 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 5 --last_pooling globalavg --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1 --filters 16 --stacks 4 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 5 --last_pooling globalavg --evaluate
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1 --filters 32 --stacks 4 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 5 --last_pooling globalavg --evaluate


PARAMS="--spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --batchnorm 1 --context_width 0 --iterations 200000 --frame_width 1024 --batch_size 16"
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 3 3 1 --filters 8 --stacks 4 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate

# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 16 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 8 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 4 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 4 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 4 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 8 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 16 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 4 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 4 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 64 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 24 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 4 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0 --logdir models/0506_151118-mir-bs16-fw1024-cw0-scqt-sus1-sos5-adeep_hcnn-f24-s8-us0-os4-cc1,1,1-lck1,1-lpglobalavg-rh2-b1-d0.0-p50459
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 48 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 40 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 4 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 80 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 48 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 4 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 96 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0

python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 8  --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 4  --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 16 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 24 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 48 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0

# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 2 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 3 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 4 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 2 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 3 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 4 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 1 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 2 --overtone_stacking 1 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 2 --overtone_stacking 2 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 2 --overtone_stacking 3 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 2 --overtone_stacking 4 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 1 1 --filters 32 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 2 --overtone_stacking 5 --last_pooling globalavg --evaluate --residual_hop 2 --dropout 0.0

PARAMS="--spectrogram cqt --architecture deep_hcnn --batchnorm 1 --context_width 0 --dropout 0.0 --residual_hop 2 --iterations 200000 --frame_width 1024 --evaluate --stacks 8 --last_conv_kernel 1 72 --last_pooling avg"
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 4 --undertone_stacking 0 --overtone_stacking 1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 12 --undertone_stacking 0 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 8 --undertone_stacking 0 --overtone_stacking 1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 24 --undertone_stacking 0 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 16 --undertone_stacking 0 --overtone_stacking 1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 48 --undertone_stacking 0 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 32 --undertone_stacking 0 --overtone_stacking 1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 96 --undertone_stacking 0 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 48 --undertone_stacking 0 --overtone_stacking 1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 144 --undertone_stacking 0 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 64 --undertone_stacking 0 --overtone_stacking 1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 144 --undertone_stacking 0 --overtone_stacking 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 72 --undertone_stacking 0 --overtone_stacking 1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 216 --undertone_stacking 0 --overtone_stacking 3

PARAMS="--spectrogram cqt --architecture deep_hcnn --batchnorm 1 --context_width 0 --dropout 0.0 --residual_hop 2 --iterations 200000 --frame_width 1024 --evaluate --stacks 8 --last_conv_kernel 1 72 --last_pooling avg --filters 32 --conv_ctx 1 --residual_end 8"
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 1 --undertone_stacking 0 --overtone_stacking 1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 1
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 2
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 3
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 4
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 5
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 6
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 8

PARAMS="--spectrogram cqt --architecture deep_hcnn --batchnorm 1 --context_width 0 --dropout 0.0 --residual_hop 2 --iterations 200000 --frame_width 1024 --evaluate --stacks 8 --last_conv_kernel 1 72 --last_pooling avg --conv_ctx 1"
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 32
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 68
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 16
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 34
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 8
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 17

# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 36
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 77 --logdir models/0515_013404-mir-fw1024-cw0-scqt-sus1-sos5-adeep_hcnn-f77-s8-us0-os1-su7-cc1-lck1,72-lpavg-rh2-b1-d0.0

# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 28
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 59

# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 24
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 50

# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 1 --overtone_stacking 5 --stacking_until 7 --filters 20
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --undertone_stacking 0 --overtone_stacking 1 --stacking_until 7 --filters 42

# ./run_kelz_comparison.sh

# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture LY --filters 32 --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 1 --batchnorm 1 --iterations 200000 --evaluate --logdir models/0514_231958-mir-sYunNingHung_cqt-sus0-sos1-aLY-f32-b1

# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture LY --filters 16 --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 1 --batchnorm 1 --iterations 200000
# python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture LY --filters 8 --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 1 --batchnorm 1 --iterations 200000




# PARAMS="--spectrogram cqt --architecture deep_hcnn --spectrogram_undertone_stacking 1 --spectrogram_overtone_stacking 5 --batchnorm 1 --context_width 0 --iterations 200000 --frame_width 1024 --batch_size 16 --evaluate"
# python -u spectrogram_mir.py $LEARNING_PARAMS2 $PARAMS --conv_ctx 1 --filters 24 --stacks 8 --last_conv_kernel 1 1 --undertone_stacking 0 --overtone_stacking 4  --stacking_until 7 --last_pooling globalavg --residual_hop 2 --dropout 0.0
