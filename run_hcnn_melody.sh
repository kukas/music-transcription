PARAMS="--frame_width 256 --batch_size 16 --annotations_per_window 10 --spectrogram_top_db 110 --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8 --undertone_stacking 5 --overtone_stacking 6 --filters 8 --stacks 4 --conv_ctx -3 -3 -3 -1 --dilations -8 -4 -2 -1 --cut_context 0 --context_width 3072 --unvoiced_loss_weight 1.0"
#LEARNING_PARAMS2="--evaluate_small_every 10000 --evaluate_every 20000 --iterations 100000 --learning_rate 0.0005 --learning_rate_decay 0.8 --learning_rate_decay_steps 20000"
# python -u spectrogram.py $PARAMS $LEARNING_PARAMS --evaluate
LOGDIR="models/0929_112608-spctrgrm-bs16-fw256-cw3072-apw10-ulw1.0-std110.0-sus8-sos8-cc0-f8-s4-us5-os6-cc-3,-3,-3,-1-d-8,-4,-2,-1"

python -u spectrogram.py $PARAMS --logdir $LOGDIR --predict $1 --output_file $2

