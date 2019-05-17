
# crepe replication
python -u crepe-normalization-melody.py --epochs 1 --datasets mdb_stem_synth --capacity_multiplier 16 --context_width 466 --bins_per_semitone 5 --annotation_smoothing 0.25 --learning_rate 0.0002

# capacity
python -u crepe_melody.py --epochs 6  --capacity_multiplier 4 --context_width 466 --bins_per_semitone 5 --annotation_smoothing 0.25 --learning_rate 0.0002
python -u crepe_melody.py --epochs 6  --capacity_multiplier 8 --context_width 466 --bins_per_semitone 5 --annotation_smoothing 0.25 --learning_rate 0.0002
python -u crepe_melody.py --epochs 6  --capacity_multiplier 16 --context_width 466 --bins_per_semitone 5 --annotation_smoothing 0.25 --learning_rate 0.0002
python -u crepe_melody.py --epochs 6  --capacity_multiplier 32 --context_width 466 --bins_per_semitone 5 --annotation_smoothing 0.25 --learning_rate 0.0002

# diskretizace
python -u crepe_melody.py --epochs 6  --capacity_multiplier 16 --context_width 466 --bins_per_semitone 1 --annotation_smoothing 0.0 --learning_rate 0.0002
python -u crepe_melody.py --epochs 6  --capacity_multiplier 16 --context_width 466 --bins_per_semitone 3 --annotation_smoothing 0.25 --learning_rate 0.0002
python -u crepe_melody.py --epochs 6  --capacity_multiplier 16 --context_width 466 --bins_per_semitone 5 --annotation_smoothing 0.25 --learning_rate 0.0002
python -u crepe_melody.py --epochs 6  --capacity_multiplier 16 --context_width 466 --bins_per_semitone 7 --annotation_smoothing 0.25 --learning_rate 0.0002
python -u crepe_melody.py --epochs 6  --capacity_multiplier 16 --context_width 466 --bins_per_semitone 9 --annotation_smoothing 0.25 --learning_rate 0.0002

# rozptyl
COMMON="--epochs 10 --capacity_multiplier 16 --context_width 2002 --multiresolution_convolution 6 --bins_per_semitone 5 --learning_rate 0.0002"
python -u crepe_melody.py  $COMMON --annotation_smoothing 0.0
python -u crepe_melody.py  $COMMON --annotation_smoothing 0.088
python -u crepe_melody.py  $COMMON --annotation_smoothing 0.124
python -u crepe_melody.py  $COMMON --annotation_smoothing 0.177
python -u crepe_melody.py  $COMMON --annotation_smoothing 0.221
python -u crepe_melody.py  $COMMON --annotation_smoothing 0.265
python -u crepe_melody.py  $COMMON --annotation_smoothing 0.354
python -u crepe_melody.py  $COMMON --annotation_smoothing 0.707

# sirka okna

python -u crepe-normalization-melody.py --epochs 10 --capacity_multiplier 8 --context_width 210  --learning_rate 0.0002
python -u crepe-normalization-melody.py --epochs 10 --capacity_multiplier 8 --context_width 466  --learning_rate 0.0002
python -u crepe-normalization-melody.py --epochs 10 --capacity_multiplier 8 --context_width 978  --learning_rate 0.0002
python -u crepe-normalization-melody.py --epochs 10 --capacity_multiplier 8 --context_width 2002  --learning_rate 0.0002
python -u crepe-normalization-melody.py --epochs 10 --capacity_multiplier 8 --context_width 4050  --learning_rate 0.0002

# multiresolution convolution

python -u crepe_melody.py --epochs 10 --capacity_multiplier 16 --batch_size 32 --multiresolution_convolution 0 --learning_rate 0.001 --learning_rate_decay 0.5 --learning_rate_decay_steps 50000 --evaluate
python -u crepe_melody.py --epochs 10 --capacity_multiplier 16 --batch_size 32 --multiresolution_convolution 2 --learning_rate 0.001 --learning_rate_decay 0.5 --learning_rate_decay_steps 50000 --evaluate
python -u crepe_melody.py --epochs 10 --capacity_multiplier 16 --batch_size 32 --multiresolution_convolution 3 --learning_rate 0.001 --learning_rate_decay 0.5 --learning_rate_decay_steps 50000 --evaluate
python -u crepe_melody.py --epochs 10 --capacity_multiplier 16 --batch_size 32 --multiresolution_convolution 4 --learning_rate 0.001 --learning_rate_decay 0.5 --learning_rate_decay_steps 50000 --evaluate
python -u crepe_melody.py --epochs 10 --capacity_multiplier 16 --batch_size 32 --multiresolution_convolution 5 --learning_rate 0.001 --learning_rate_decay 0.5 --learning_rate_decay_steps 50000 --evaluate
python -u crepe_melody.py --epochs 10 --capacity_multiplier 16 --batch_size 32 --multiresolution_convolution 6 --learning_rate 0.001 --learning_rate_decay 0.5 --learning_rate_decay_steps 50000 --evaluate
python -u crepe_melody.py --epochs 10 --capacity_multiplier 16 --batch_size 32 --multiresolution_convolution 7 --learning_rate 0.001 --learning_rate_decay 0.5 --learning_rate_decay_steps 50000 --evaluate
python -u crepe_melody.py --epochs 10 --capacity_multiplier 16 --batch_size 32 --multiresolution_convolution 8 --learning_rate 0.001 --learning_rate_decay 0.5 --learning_rate_decay_steps 50000 --evaluate

