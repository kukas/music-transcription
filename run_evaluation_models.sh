set -xe

# === CREPE MODEL ===
PARAMS="--capacity_multiplier 8 --multiresolution_convolution 5"
LOGDIR="models/0506_222602-crepe-bs32-lr0.001-lrd0.75-lrds50000-cm8-mc5-vsFalse/"
# Train the model
# LEARNING_PARAMS="--epochs 5 --batch_size 32 --learning_rate 0.001 --learning_rate_decay 0.75 --learning_rate_decay_steps 50000"
# python -u crepe_melody.py $PARAMS $LEARNING_PARAMS --logdir $LOGDIR
# Evaluate the model on MIREX datasets
python -u crepe_melody.py $PARAMS --evaluate --logdir $LOGDIR --dataset mirex05 orchset adc04
# Evalute on the rest of the data
# python -u crepe_melody.py $PARAMS --evaluate --logdir $LOGDIR --dataset mdb
# python -u crepe_melody.py $PARAMS --evaluate --logdir $LOGDIR --dataset mdb_melody_synth
# python -u crepe_melody.py $PARAMS --evaluate --logdir $LOGDIR --dataset wjazzd

# === WAVENET MODEL ===
PARAMS="--filter_width 3 --initial_filter_width 0 --max_dilation 1024 --stack_number 2 --residual_channels 16 --skip_channels 16 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160  --skip concat --postprocessing avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
LOGDIR="models/0501_040822-wavenet-fw160-cw8192-apw5-lrd0.8-lrds10000-ifw0-ifpsame-fw3-ubTrue-sc16-rc16-sn2-md1024-dld0.0-sld0.0-sconcat-pavgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
# Train the model
# LEARNING_PARAMS="--iterations 150000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --evaluate_every 20000"
# python -u wavenet.py $PARAMS $LEARNING_PARAMS --logdir $LOGDIR
# Evaluate the model on MIREX datasets
python -u wavenet.py $PARAMS --evaluate --logdir $LOGDIR --dataset mirex05 orchset adc04
# Evalute on the rest of the data
# python -u wavenet.py $PARAMS --evaluate --logdir $LOGDIR --dataset mdb
# python -u wavenet.py $PARAMS --evaluate --logdir $LOGDIR --dataset mdb_melody_synth
# python -u wavenet.py $PARAMS --evaluate --logdir $LOGDIR --dataset wjazzd

# === HCNN no context ===
PARAMS="--frame_width 256 --annotations_per_window 1 --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8 --filters 16 --stacks 8 --undertone_stacking 2 --overtone_stacking 3"
LOGDIR="models/0513_173718-spctrgrm-bs16-fw256-apw1-ulw1.0-sus8-sos8-f16-s8-us2-os3"
# Train the model
# LEARNING_PARAMS="--evaluate_small_every 10000 --evaluate_every 10000 --iterations 100000"
# python -u spectrogram.py $PARAMS $LEARNING_PARAMS --logdir $LOGDIR
# Evaluate the model on MIREX datasets
python -u spectrogram.py $PARAMS --evaluate --logdir $LOGDIR --dataset mirex05 orchset adc04
# Evalute on the rest of the data
# python -u spectrogram.py $PARAMS --evaluate --logdir $LOGDIR --dataset mdb
# python -u spectrogram.py $PARAMS --evaluate --logdir $LOGDIR --dataset mdb_melody_synth
# python -u spectrogram.py $PARAMS --evaluate --logdir $LOGDIR --dataset wjazzd

# === HCNN context ===
PARAMS="--frame_width 256 --batch_size 8 --annotations_per_window 10 --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8 --undertone_stacking 1 --overtone_stacking 2 --filters 16 --stacks 4 --spectrogram cqt_fs --specaugment_prob 0.75 --conv_ctx -3 -3 -3 -1 --dilations -8 -4 -2 -1 --cut_context 0 --context_width 3072 --unvoiced_loss_weight 1.0"
LOGDIR="models/0513_114331-spctrgrm-bs8-fw256-cw3072-apw10-ulw1.0-scqt_fs-sus8-sos8-cc0-f16-s4-us1-os2-cc-3,-3,-3,-1-d-8,-4,-2,-1-sp0.75"
# Train the model
# LEARNING_PARAMS="--evaluate_small_every 10000 --evaluate_every 10000 --iterations 100000 --stop_if_too_slow 100"
# python -u spectrogram.py $PARAMS $LEARNING_PARAMS --logdir $LOGDIR
# Evaluate the model on MIREX datasets
python -u spectrogram.py $PARAMS --evaluate --dataset orchset adc04 mirex05 --logdir $LOGDIR
# Evalute on the rest of the data
# python -u spectrogram.py $PARAMS --evaluate --dataset mdb --logdir $LOGDIR
# python -u spectrogram.py $PARAMS --evaluate --dataset wjazzd --logdir $LOGDIR
# python -u spectrogram.py $PARAMS --evaluate --dataset mdb_melody_synth --logdir $LOGDIR
