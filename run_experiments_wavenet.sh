set -xe

# BASELINE Martak
wavenet.py --evaluate --filter_width 2 --initial_filter_width 2 --max_dilation 512 --stack_number 2 --residual_channels 128 --skip_channels 128 --batch_size 20 --min_note 0 --note_range 128 --bins_per_semitone 1 --annotation_smoothing 0 --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 2100 --use_biases --frame_width 160 --skip add --postprocessing conv_f128_k1_s1_Psame_arelu--conv_f128_k1_s1_Psame--avgpool_p160_s160_Psame
wavenet.py --evaluate --filter_width 2 --initial_filter_width 2 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 --batch_size 8 --min_note 0 --note_range 128 --bins_per_semitone 1 --annotation_smoothing 0 --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 2100 --use_biases --frame_width 160 --skip add --postprocessing conv_f128_k1_s1_Psame_arelu--conv_f128_k1_s1_Psame--avgpool_p160_s160_Psame
wavenet.py --evaluate --filter_width 2 --initial_filter_width 2 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 2100 --use_biases --frame_width 160 --skip add --postprocessing conv_f360_k1_s1_Psame_arelu--conv_f360_k1_s1_Psame--avgpool_p160_s160_Psame

# Vliv počtu kanálů
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 4 --skip_channels 4 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 8 --skip_channels 8 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 24 --skip_channels 24 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 32 --skip_channels 32 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 40 --skip_channels 40 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 48 --skip_channels 48 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

# Gridsearch přes dilations a nb_stacks
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 64 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 64 --stack_number 2 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 64 --stack_number 3 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 64 --stack_number 4 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 128 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 128 --stack_number 2 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 128 --stack_number 3 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 128 --stack_number 4 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 256 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 256 --stack_number 2 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 256 --stack_number 3 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 256 --stack_number 4 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 2 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 3 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 4 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 1024 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 1024 --stack_number 2 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 1024 --stack_number 3 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 1024 --stack_number 4 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 2048 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

##################### TOTO JEŠTĚ KLIDNĚ NECHAT DOBĚHNOUT #############################
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 2048 --stack_number 2 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame" --rewind --logdir "models/0504_133139-wavenet-fw160-cw8192-apw5-lrd0.8-lrds10000-ifw0-ifpsame-fw3-ubTrue-sc16-rc16-sn2-md2048-dld0.0-sld0.0-sconcat-pavgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 2048 --stack_number 3 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 4096 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 4096 --stack_number 2 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
##################### TOTO JEŠTĚ KLIDNĚ NECHAT DOBĚHNOUT #############################


# Vliv velikosti šířky kernelu dilatací
python -u wavenet.py --evaluate --filter_width 2 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 4 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"
python -u wavenet.py --evaluate --filter_width 5 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

# Vliv redukce skip propojení

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip add --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

# ## Vliv velikosti první konvoluce
python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 8 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 16 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 32 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 64 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 256 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 1024 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip last --postprocessing "avgpool_p5_s5_Psame--conv_f180_k16_s4_Psame_arelu--conv_f360_k64_s8_Psame"


# Vliv velikosti posledních vrstev

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p160_s160_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f360_k64_s32_Psame"


python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "conv_f360_k80_s160_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "conv_f360_k160_s160_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "conv_f360_k320_s160_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "conv_f720_k320_s160_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f360_k32_s32_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f360_k128_s32_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p5_s5_Psame--conv_f360_k256_s32_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p10_s10_Psame--conv_f360_k16_s16_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p10_s10_Psame--conv_f360_k32_s16_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p10_s10_Psame--conv_f360_k64_s16_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p20_s20_Psame--conv_f360_k8_s8_Psame"

python -u wavenet.py --evaluate --filter_width 2 --initial_filter_width 2 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 2100 --use_biases --frame_width 160 \
    --skip add --postprocessing "conv_f360_k1_s1_Psame_arelu--conv_f360_k1_s1_Psame--avgpool_p160_s160_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p20_s20_Psame--conv_f360_k16_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p20_s20_Psame--conv_f360_k32_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p20_s20_Psame--conv_f360_k64_s8_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p160_s160_Psame--conv_f360_k1_s1_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p160_s160_Psame--conv_f360_k3_s1_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p160_s160_Psame--conv_f360_k5_s1_Psame"

python -u wavenet.py --evaluate --filter_width 3 --initial_filter_width 0 --max_dilation 512 --stack_number 1 --residual_channels 16 --skip_channels 16 \
    --iterations 100000 --learning_rate_decay_steps 10000 --learning_rate_decay 0.8 --annotations_per_window 5 --context_width 8192 --use_biases --frame_width 160 \
    --skip concat --postprocessing "avgpool_p160_s160_Psame--conv_f360_k5_s1_Psame"
