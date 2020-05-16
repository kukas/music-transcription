set -xe
HCQT="--spectrogram_undertone_stacking 0 --spectrogram_overtone_stacking 4"
HConv="--undertone_stacking 1 --overtone_stacking 3"
LEARNING_PARAMS="--stop_if_too_slow 60 --threads 6 --evaluate_small_every 5000 --evaluate_every 20000 --iterations 100000 --evaluate"
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 4
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 4 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 4 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 8
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 8 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 8 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 12
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 12 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 12 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 16
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 16 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 16 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 20
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 20 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 20 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 24
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 24 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 24 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 28
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 28 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 28 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 32
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 32 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 32 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 36
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 36 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 40
python -u kelz.py $LEARNING_PARAMS --architecture allconv --filters 40 $HCQT

LEARNING_PARAMS="--stop_if_too_slow 60 --threads 6 --evaluate_small_every 5000 --evaluate_every 20000 --iterations 150000 --evaluate"

python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 8
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 8 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 8 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 12
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 12 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 12 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 16
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 16 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 16 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 20
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 20 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 20 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 24
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 24 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 24 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 28
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 28 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 28 $HCQT $HConv
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 32
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 32 $HCQT
python -u kelz.py $LEARNING_PARAMS --architecture vggnet --filters 32 $HCQT $HConv
