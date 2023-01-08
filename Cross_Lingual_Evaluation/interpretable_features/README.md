Code to reproduce the experiments related with interpretable features. Interpretable features include: prosodic, linguistic and cognitive descriptors.

The pipeline goes as follows:

1) Data pre-processing
  - To resample the speech recordings to 16 kHz run the script ```convert_to_16k.sh```
  - To extract speech transcripts from speech recordings using Whisper run ```/extract_transcripts.py```

2) Features extraction 

    - To extract the acoustic features described in the paper run ```feature_extraction/extract_acoustic_feats.sh```
    - To extract the linguistic features described in the paper run ```feature_extraction/extract_linguistic_feats.py``` 
    - To extract the cognitive features described in the paper run ```feature_extraction/extract_cognitive_feats.py```
   
3) Grid search

    - To identify the best hyperparameters for each model within the mono-lingual and multi-lingual experiments run ```grid_search/run_gridsearch.sh```.
   
5) Classification 

   - run ```monolingual_experiments.sh``` for mono-lingual experiments
   - run ```multilingual_experiments.sh``` for cross-lingual experiments 
   - run ```crosslingual_experiments.sh``` for cross-lingual experiments

   All the experiments are organized separately for each language. In doing so, we also evaluated different tasks separately. The three tasks evaluated were a reading passage (RP), a spontaneous speech (SS) task, and text dependent utterances (TDU).