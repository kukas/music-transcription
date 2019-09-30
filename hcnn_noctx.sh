PARAMS="--frame_width 256 --annotations_per_window 1 --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8 --unvoiced_loss_weight 1.0  --filters 12 --stacks 8 --undertone_stacking 3 --overtone_stacking 5 --spectrogram cqt --spectrogram_top_db 110"
# LEARNING_PARAMS="--evaluate_small_every 10000 --evaluate_every 20000 --iterations 200000 --learning_rate 0.001 --learning_rate_decay 0.8 --learning_rate_decay_steps 20000"
# python -u spectrogram.py $PARAMS $LEARNING_PARAMS
LOGDIR="models/0926_024712-spctrgrm-fw256-apw1-lr0.0005-ulw1.0-scqt-std110.0-sus8-sos8-f12-s8-us3-os5"
python -u spectrogram.py $PARAMS --logdir $LOGDIR --predict $1 --output_file $2
