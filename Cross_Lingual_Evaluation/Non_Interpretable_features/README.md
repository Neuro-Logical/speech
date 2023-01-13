## Non-interpretable approaches: x-vectors and TRILLsson
Before running the code, install the dependencies from `requirements.txt`

The whole experiment pipeline can be run through `run_all_experiments.sh`. X-vectors and TRILLsson representations are first extracted and saved in the feature directory. After that, mono-lingual, multi-lingual, and cross-lingual experiments will be run.

Please set the paths of feature directory and output result directory before running the script. 
The feature folder needs to contain sub-folders *xvector* and *trill*. Access to the JHU CLSP grid is required to run the experiments with the data sets and data partitions that match the ones we used in the interpretable approaches. All experiment output results will be saved in the output result directory.
