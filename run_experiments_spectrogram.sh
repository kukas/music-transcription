EVAL_BASICS="--evaluate_small_every 10000 --evaluate_every 10000 --iterations 100000"
DIL_BASICS="--frame_width 256 --batch_size 8 --annotations_per_window 10 --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8"
DIL_SPEC="$EVAL_BASICS $DIL_BASICS --undertone_stacking 1 --overtone_stacking 2"

# harmonické transformace

python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 0 --overtone_stacking 1 --filters 8  --stacks 4
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 0 --overtone_stacking 1 --filters 16 --stacks 4
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 0 --overtone_stacking 1 --filters 32 --stacks 4
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 0 --overtone_stacking 1 --filters 8  --stacks 8
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 0 --overtone_stacking 1 --filters 16 --stacks 8
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 0 --overtone_stacking 1 --filters 32 --stacks 8
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 0 --overtone_stacking 1 --filters 8  --stacks 16
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 0 --overtone_stacking 1 --filters 16 --stacks 16
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 0 --overtone_stacking 1 --filters 32 --stacks 16

python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 1 --overtone_stacking 2 --filters 8  --stacks 4
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 1 --overtone_stacking 2 --filters 16 --stacks 4
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 1 --overtone_stacking 2 --filters 32 --stacks 4
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 1 --overtone_stacking 2 --filters 8  --stacks 8
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 1 --overtone_stacking 2 --filters 16 --stacks 8
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 1 --overtone_stacking 2 --filters 32 --stacks 8
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 1 --overtone_stacking 2 --filters 8  --stacks 16
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 1 --overtone_stacking 2 --filters 16 --stacks 16
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 1 --overtone_stacking 2 --filters 32 --stacks 16

python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 2 --overtone_stacking 3 --filters 8  --stacks 4
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 2 --overtone_stacking 3 --filters 16 --stacks 4
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 2 --overtone_stacking 3 --filters 32 --stacks 4
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 2 --overtone_stacking 3 --filters 8  --stacks 8
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 2 --overtone_stacking 3 --filters 16 --stacks 8
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 2 --overtone_stacking 3 --filters 32 --stacks 8
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 2 --overtone_stacking 3 --filters 8  --stacks 16
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 2 --overtone_stacking 3 --filters 16 --stacks 16
python -u spectrogram.py $EVAL_BASICS $DIL_BASICS --undertone_stacking 2 --overtone_stacking 3 --filters 32 --stacks 16

# Parametr hop_size

TEST_SPEC="$EVAL_BASICS --batch_size 16 --context_width 0 --annotations_per_window 1 --spectrogram_undertone_stacking 8 --spectrogram_overtone_stacking 8  --undertone_stacking 2 --overtone_stacking 3"

python -u spectrogram.py $TEST_SPEC --filters 8  --stacks 4 --frame_width 256
python -u spectrogram.py $TEST_SPEC --filters 16 --stacks 4 --frame_width 256
python -u spectrogram.py $TEST_SPEC --filters 32 --stacks 4 --frame_width 256
python -u spectrogram.py $TEST_SPEC --filters 8  --stacks 8 --frame_width 256
python -u spectrogram.py $TEST_SPEC --filters 16 --stacks 8 --frame_width 256
python -u spectrogram.py $TEST_SPEC --filters 32 --stacks 8 --frame_width 256
python -u spectrogram.py $TEST_SPEC --filters 8  --stacks 16 --frame_width 256
python -u spectrogram.py $TEST_SPEC --filters 16 --stacks 16 --frame_width 256
python -u spectrogram.py $TEST_SPEC --filters 32 --stacks 16 --frame_width 256

python -u spectrogram.py $TEST_SPEC --filters 8  --stacks 4 --frame_width 512
python -u spectrogram.py $TEST_SPEC --filters 16 --stacks 4 --frame_width 512
python -u spectrogram.py $TEST_SPEC --filters 32 --stacks 4 --frame_width 512
python -u spectrogram.py $TEST_SPEC --filters 8  --stacks 8 --frame_width 512
python -u spectrogram.py $TEST_SPEC --filters 16 --stacks 8 --frame_width 512
python -u spectrogram.py $TEST_SPEC --filters 32 --stacks 8 --frame_width 512
python -u spectrogram.py $TEST_SPEC --filters 8  --stacks 16 --frame_width 512
python -u spectrogram.py $TEST_SPEC --filters 16 --stacks 16 --frame_width 512
python -u spectrogram.py $TEST_SPEC --filters 32 --stacks 16 --frame_width 512

# vícekanálový vstup CQT
python -u spectrogram.py $DIL_SPEC --filters 8 --stacks 4 --spectrogram cqt_fs
python -u spectrogram.py $DIL_SPEC --filters 16 --stacks 4 --spectrogram cqt_fs
python -u spectrogram.py $DIL_SPEC --filters 32 --stacks 4 --spectrogram cqt_fs
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 8 --spectrogram cqt_fs
python -u spectrogram.py $DIL_SPEC --filters 16 --stacks 8 --spectrogram cqt_fs
python -u spectrogram.py $DIL_SPEC --filters 32 --stacks 8 --spectrogram cqt_fs
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 16 --spectrogram cqt_fs
python -u spectrogram.py $DIL_SPEC --filters 16 --stacks 16 --spectrogram cqt_fs
python -u spectrogram.py $DIL_SPEC --filters 32 --stacks 16 --spectrogram cqt_fs

# Context

python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 4 --conv_ctx 3
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 4 --conv_ctx 9 7 5 1
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 4 --conv_ctx -9 -7 -5 -1
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 4 --conv_ctx -3 -1 --dilations -2 -1 --cut_context 0 --context_width 512
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 4 --conv_ctx -3 -3 -1 --dilations -2 -2 -1 --cut_context 0 --context_width 1024
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 4 --conv_ctx -3 -3 -3 -1 --dilations -2 -2 -2 -1 --cut_context 0 --context_width 1536
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 4 --conv_ctx -3 -3 -1 --dilations -4 -2 -1 --cut_context 0 --context_width 1536
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 4 --conv_ctx -3 -3 -3 -1 --dilations -8 -4 -2 -1 --cut_context 0 --context_width 3072


python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 4 --conv_ctx 3
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 4 --conv_ctx 9 7 5 1
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 4 --conv_ctx -9 -7 -5 -1
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 4 --conv_ctx -3 -1 --dilations -2 -1 --cut_context 0 --context_width 512
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 4 --conv_ctx -3 -3 -1 --dilations -2 -2 -1 --cut_context 0 --context_width 1024
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 4 --conv_ctx -3 -3 -3 -1 --dilations -2 -2 -2 -1 --cut_context 0 --context_width 1536
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 4 --conv_ctx -3 -3 -1 --dilations -4 -2 -1 --cut_context 0 --context_width 1536
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 4 --conv_ctx -3 -3 -3 -1 --dilations -8 -4 -2 -1 --cut_context 0 --context_width 3072


python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 8 --conv_ctx 3
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 8 --conv_ctx 9 7 5 1
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 8 --conv_ctx -9 -7 -5 -1
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 8 --conv_ctx -3 -1 --dilations -2 -1 --cut_context 0 --context_width 512
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 8 --conv_ctx -3 -3 -1 --dilations -2 -2 -1 --cut_context 0 --context_width 1024
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 8 --conv_ctx -3 -3 -3 -1 --dilations -2 -2 -2 -1 --cut_context 0 --context_width 1536
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 8 --conv_ctx -3 -3 -1 --dilations -4 -2 -1 --cut_context 0 --context_width 1536
python -u spectrogram.py $DIL_SPEC --filters 8  --stacks 8 --conv_ctx -3 -3 -3 -1 --dilations -8 -4 -2 -1 --cut_context 0 --context_width 3072


python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 8 --conv_ctx 3
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 8 --conv_ctx 9 7 5 1
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 8 --conv_ctx -9 -7 -5 -1
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 8 --conv_ctx -3 -1 --dilations -2 -1 --cut_context 0 --context_width 512
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 8 --conv_ctx -3 -3 -1 --dilations -2 -2 -1 --cut_context 0 --context_width 1024
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 8 --conv_ctx -3 -3 -3 -1 --dilations -2 -2 -2 -1 --cut_context 0 --context_width 1536
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 8 --conv_ctx -3 -3 -1 --dilations -4 -2 -1 --cut_context 0 --context_width 1536
python -u spectrogram.py $DIL_SPEC --filters 16  --stacks 8 --conv_ctx -3 -3 -3 -1 --dilations -8 -4 -2 -1 --cut_context 0 --context_width 3072

