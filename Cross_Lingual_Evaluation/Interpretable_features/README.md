Code to reproduce the experiments related with interpretable features. Interpretable features include: prosodic, linguistic and cognitive descriptors.

The pipeline goes as follows:

1) Data pre-processing

  - To resample the speech recordings to 16 kHz run the script ```convert_to_16k.sh```
  - To extract speech transcripts from speech recordings using Whisper run ```extract_transcripts.py```

2) Features extraction 

    - To extract the acoustic features described in the paper run ```feature_extraction/extract_acoustic_feats.sh```
    - To extract the linguistic features described in the paper run ```feature_extraction/extract_linguistic_feats.py``` .
    - To extract the linguistic and cognitive features from the transcripts of your speech data set, you can use the notebook: ```feature_extraction/extract_ling_cog_features.ipynb```
    - To extract the cognitive features described in the paper run ```feature_extraction/extract_cognitive_feats.py```
   
3) Grid search

    - To identify the best hyperparameters for each model within the mono-lingual and multi-lingual experiments run ```grid_search/run_gridsearch.sh```.
   
4) Classification 

   - run ```monolingual_experiments.sh``` for mono-lingual experiments
   - run ```multilingual_experiments.sh``` for cross-lingual experiments 
   - run ```crosslingual_experiments.sh``` for cross-lingual experiments

   All the experiments are organized separately for each language. In doing so, we also evaluated different tasks separately. The three tasks evaluated were a reading passage (RP), a spontaneous speech (SS) task, and text dependent utterances (TDU).


### Prepare data 
For the classification experiments, the following data structure is required for each language. For each family of features (i.e., acoustic, cognitive, linguistic), create a csv file structured as reported in the table below.

| Column name      | Data Type | Description  |
| :---        |    :----:   |          ---: |
| Feature name      | int or float  |Extracted feature value |
| Speaker ID | string | Unique identifier of each subject that should contain information about the class (i.e., HC, PD),  <br> the subject ID (e.g., 12), and the the task (e.g., monologue). An exaple is: HC_12_monologue.wav | 

A mock csv file reproducing the structure above is reported in ``` data/data_interpretable_feats.csv ```.
