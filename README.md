# Better Framewise Learning over Spectrograms with Harmonic Convolutions

A small python framework for conducting deep learning experiments for framewise MIR tasks. The aim of this project is to create a set of utility functions and classes that make the experiments easier to implement and replicate. A sample set of experiments is included. 

## Installation

To run scripts in this repository, you need Python 3 with additional dependencies specified in `requirements.txt`. You can either install the dependencies system-wide or use the included Anaconda environment file:

### a) Install the Anaconda environment (Recommended)

    conda env create -f environment.yml
    conda activate melody_extraction

Important: if you own a GPU compatible with Tensorflow, install `tensorflow-gpu`:

    conda install tensorflow-gpu

### b) Installing the dependencies using pip

    pip install -r requirements.txt

## Usage

You can also check and replicate the results we reported in our paper.

### Replicate our experiments

To replicate our experiments, download the data, train your models and evaluate them.

#### Prepare the data

For training we used [MedleyDB dataset](https://medleydb.weebly.com/) which is available after a permission request. MedleyDB is offered free of charge for non-commercial research use.
- After obtaining the dataset, extract it to `data/` directory.
We also used MAPS and MusicNet (zip package). Extract them to `data/` directory.

#### Run the training and evaluations

For running the training and evaluation see the scripts `run_*.sh`.

## Dataset splits

As noted above, dataset splits used are located in `data/`. MedleyDB and MDB-melody-synth split is in file `mdb_ismir_split.json`, MusicNet split is in `musicnet_my_split.json`, MAPS split is in `maps_kelz_split.json`.

## Features of the framework
- Dataset handling
    - loading common AME datasets (MedleyDB, MDB-melody-synth, ORCHSET, ...)
        - Some AMT datasets are also supported (MDB-mf0-synth)
    - using training, validation and testing splits
    - audio preprocessing
        - audio file resampling
        - annotation resampling to different hop sizes
        - spectrogram generations
        - preprocessing operations are parallelized and all the results cached to the disk and automatically recomputed when the parameters change
    - simple manipulation of the audio and spectrograms for creating small evaluation excerpts (for qualitative analysis)
        - annotations of the excerpts are automatically matched to the full annotations
    - using datasets for training deep neural networks
        - overlapping windows, batching, shuffling
        - one example window can contain more annotation points
        - data filtering by voicing
        - data augmentation using [SpecAugment](https://arxiv.org/abs/1904.08779)

- Tensorflow model skeleton
    - the focus is on replicability of the experiments, all of the parameters of the experiments are visible in Tensorboard and a subset is also automatically added to the name of the experiment.
        - also the model function is saved alongside the experiment files
    - a set of common functions
        - input audio normalization
        - target annotation representation
            - one-hot or gaussian blurred output (as in [2])
    - training loop
        - periodic qualitative and quantitative evaluation, 
        - saving of the best models

- Evaluation
    - everything is logged to Tensorboard
        - images of the output salience and prediction
        - MIREX statistics

- Visualization
    - for use in Jupyter notebooks and in Tensorboard
    - colorful pianoroll with the reference and estimate annotation
    - confusion matrix
    - note prediction distance histogram

And more...

## Citations and Acknowledgements

> [1] Salamon, J., & Gomez, E. (2012). Melody extraction from polyphonic music signals using pitch contour characteristics. IEEE Transactions on Audio, Speech and Language Processing, 20(6), 1759–1770. https://doi.org/10.1109/TASL.2012.2188515

> [2] Bittner, R. M., Mcfee, B., Salamon, J., Li, P., & Bello, J. P. (2017). Deep Salience Representations for F0 Estimation in Polyphonic Music. Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR 2017), 63–70. Retrieved from https://bmcfee.github.io/papers/ismir2017_salience.pdf

> [3] Basaran, D., Essid, S., & Peeters, G. (2018). Main Melody Estimation with Source-Filter NMF and CRNN. Proceedings of the 19th International Society for Music Information Retrieval Conference, 82–89. https://doi.org/10.5281/zenodo.1492349

We would also like to thank the Jazzomat Research Project for kindly providing the WJazzD testing data.
