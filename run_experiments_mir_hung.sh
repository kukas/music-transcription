set -x

PARAMS="--spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8"
LEARNING_PARAMS2="--stop_if_too_slow 60 --threads 6 --evaluate_small_every 5000 --evaluate_every 50000 --evaluate --iterations 200000"

python -u spectrogram_mir.py $LEARNING_PARAMS2 --spectrogram YunNingHung_cqt --architecture LY --spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 0 --batchnorm 1 --class_weighting 0