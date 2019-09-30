# Audio Melody Extraction

A small python framework for conducting deep learning experiments for the [MIREX AME task](https://www.music-ir.org/mirex/wiki/2019:Audio_Melody_Extraction). The aim of this project is to create a set of utility functions and classes that make the experiments easier to implement and replicate. A sample set of experiments is included. The classes were implemented to be reusable for the [Multiple Fundamental Frequency Estimation Tracking task](https://www.music-ir.org/mirex/wiki/2019:Multiple_Fundamental_Frequency_Estimation_%26_Tracking).

Using this framework we achieved state-of-the-art performance on most publicly available Melody Extraction datasets. You can read more about the [Harmonic Convolutional Neural Network in our extended abstract](https://www.music-ir.org/mirex/abstracts/2019/BH1.pdf) for MIREX 2019. 

Method       | ADC04     | MDB-melody-synth test set | MIREX05 training set | MedleyDB test set | ORCHSET   | WJazzD test set
------------ | --------- | ------------------------- | -------------------- | ----------------- | --------- | ---------------
   SAL [1]   |   0.714   |                   0.527   |              0.715   |         0.519     |   0.235   |         0.667
   BIT [2]   |   0.716   |                   0.633   |              0.702   |         0.611     |   0.407   |         0.692
   BAS [3]   |   0.669   |                 **0.689** |            **0.734** |         0.640     |   0.483   |         0.700
*HCNN noctx* |   0.735   |                   0.636   |              0.728   |         0.645     |   0.457   |       **0.714**
*HCNN ctx*   | **0.746** |                   0.637   |              0.704   |       **0.646**   | **0.525** |         0.711

You can find the list of tracks included in the test sets in `data/` directory. CSV outputs of the algorithms that were used to compute this table can be found [here](http://jirkabalhar.cz/melody-outputs.zip)

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

There are several possible uses of this github project. You can use the pretrained models for melody extraction on a specified input song. You can also check and replicate the results we reported in our paper and my [bachelor thesis on Melody Extraction](http://hdl.handle.net/20.500.11956/108322). Finally you can use our framework to train your own models.

### Run the pretrained models on a input WAV file

To run the pretrained models you need to download them.

#### Prepare the pretrained models

[Download the models](http://jirkabalhar.cz/melody_extraction_models_mirex.zip) and extract them to `models/` directory. The archive contains:
- **HCNN model (no context)**
- **HCNN model (with context)**

You can also [download the models used in my bachelor thesis](http://jirkabalhar.cz/melody_extraction_models.zip). This archive additionaly contains:
- CREPE inspired model
- WaveNet inspired model

#### Running pretrained HCNN on a specified file

The HCNN ctx and HCNN noctx models can be run using:

    ./hcnn_ctx.sh INPUT_WAV OUTPUT_CSV

and

    ./hcnn_noctx.sh INPUT_WAV OUTPUT_CSV


### Replicate our experiments

To replicate the experiments on the available MIREX datasets please [Prepare the pretrained models](#prepare-the-pretrained-models). After that you need to prepare the testing data.

#### Prepare the data

- Download the evaluation datasets:
    - [ORCHSET](https://www.upf.edu/web/mtg/orchset)
    - [MIREX05 and ADC04](https://labrosa.ee.columbia.edu/projects/melody/) (links "adc2004_full_set.zip" and "Zip file of the LabROSA training data")
- Extract the contents to `data/` directory.

For training we used [MedleyDB dataset](https://medleydb.weebly.com/) which is available after a permission request. MedleyDB is offered free of charge for non-commercial research use.
- After obtaining the dataset, extract it to `data/` directory.

#### Run the evaluations

For running the evaluations see `run_evaluation_models.sh`.

### Training custom models

MedleyDB dataset is recommended for training the models. Example training commands are included as comments in `run_evaluation_models.sh`. Also it should be noted, that running the training on a Tensorflow-compatible GPU is almost mandatory. Provided models were trained on NVIDIA GTX 1070. All the experiments included in the bachelor thesis can be replicated using the scripts `run_experiments_*.sh`.


## Dataset splits

As noted above, dataset splits used in the MIREX extended abstract and my bachelor thesis are located in `data/`. MedleyDB and MDB-melody-synth split is in file `mdb_ismir_split.json`, WJazzD split is in `wjazzd_split.json`.

## Disclaimer

This framework was created for use in my bachelor thesis. Since the focus of the thesis wasn't on the code but on the results of the experiments, this repository doesn't contain the best practices. Also with the release of Tensorflow r2 a part of the code is deprecated now and I am most probably not going to refactor it. Still a lot of the code might be useful to someone and the experiments are made to be easily replicable.

## Features
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
