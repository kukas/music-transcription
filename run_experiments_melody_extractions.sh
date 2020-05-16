set -xe

HCNN="--undertone_stacking 2 --overtone_stacking 3"
PARAMS="--batch_size 8  --evaluate_every 2500 --iterations 30000"
python -u bittner.py --capacity_multiplier 4 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 4 $PARAMS $HCNN --evaluate
python -u bittner.py --capacity_multiplier 6 $PARAMS $HCNN --evaluate
python -u bittner.py --capacity_multiplier 8 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 8 $PARAMS $HCNN --evaluate
python -u bittner.py --capacity_multiplier 10 $PARAMS $HCNN --evaluate
python -u bittner.py --capacity_multiplier 12 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 12 $PARAMS $HCNN --evaluate
python -u bittner.py --capacity_multiplier 14 $PARAMS $HCNN --evaluate
python -u bittner.py --capacity_multiplier 16 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 16 $PARAMS $HCNN --evaluate
python -u bittner.py --capacity_multiplier 18 $PARAMS $HCNN --evaluate
python -u bittner.py --capacity_multiplier 20 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 22 $PARAMS $HCNN --evaluate
python -u bittner.py --capacity_multiplier 24 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 28 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 32 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 36 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 40 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 44 $PARAMS --evaluate
python -u bittner.py --capacity_multiplier 48 $PARAMS --evaluate