## Non-interpretable approaches: x-vectors, TRILLsson, Wav2Vec2, and HuBERT
Before running the code, install the dependencies from `requirements.txt`

The whole experiment pipeline can be run through `run_all_experiments.sh`. X-vectors, TRILLsson, Wav2Vec2, and HuBERT representations are first extracted and saved in the feature directory. After that, mono-lingual, multi-lingual, and cross-lingual experiments will be run.

Please set the paths of feature directory and output result directory before running the script. Access to the JHU CLSP grid is required to run the experiments with the data sets and data partitions that match the ones we used in the interpretable approaches. All experiment output results will be saved in the output result directory.
