# Automatic Music Transcription

A small framework for conducting deep learning experiments for the [MIREX AMT task](http://www.music-ir.org/mirex/wiki/2018:Multiple_Fundamental_Frequency_Estimation_%26_Tracking). The aim of this project is to create a set of utility functions and classes that make the experiments easier to implement and replicate. A sample set of experiments is included. The classes should be reusable for [Melody Extraction task](http://www.music-ir.org/mirex/wiki/2018:Audio_Melody_Extraction) since melody can be viewed as a monophonic subset of a complete transcription of audio.

![piano roll](docs/pianoroll.png)

## Features
- Dataset handling
    - [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html) dataset loading and automatic resampling. Other datasets such as MIREX 2007 MultiF0, Bach10, Su, MedleyDB are not supported yet but should be easy to add.
    - Slicing of the dataset audio for creating small testing subsets.
    - Automatic processing of a dataset - spectrogram precomputation and management
- Visualization tools for examining the model output.
    - Piano roll for comparison between the _gold truth_ and _estimation_
    - Interactive audio output for Jupyter notebooks
    - STFT and constant-Q spectrograms (using `librosa`)
- Tensorflow model skeleton
    - Training, evaluation and inference functions
    - Detailed evaluation summary in Tensorboard
        - Evaluation of the testing set using `mir_eval`, implementation of basic metrics in Tensorflow for training information
        - Visual qualitative example = piano roll of a transcription
    - Saving the model weights and topology

## Usage
The framework is intended for use with [Jupyter](http://jupyter.org/) but majority of the functions are usable also as standard modules. Only the `visualization` module might fail to run outside of interactive context.
Examples can be found in this repository but the structure of the experiments stays the same:
- import the modules
- load data
- create datasets
- define a network topology
- construct the network
- train the network
- optionally evaluate the network in the Jupyter notebook

## Dependencies
- `tensorflow`
- `numpy`
- `scipy`
- `matplotlib`
- `soundfile`
- `resampy`
- `intervaltree`
- `csv`
- `mir_eval`
- `librosa`
