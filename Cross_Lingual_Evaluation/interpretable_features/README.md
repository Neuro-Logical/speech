Code to reproduce the experiments related to the interpretable features.
The pipeline goes as follows:
- Data pre-processing
  - To resample the speech recordings to 16 kHZ run the script ***convert_to_16k.sh***
  - To extract speech transcripts from speech recordings using Whisper run ***extract_transcripts.py***
- Features extraction 
  - To extract the acoustic features described in the paper run ***extract_acoustic_feats.sh***. 
  - To extract the linguistic features described in the paper run ***extract_linguistic_feats.py***. 
  - To extract the cognitive features described in the paper run ***extract_cognitive_feats.py***. 
- Grid search;
- Nested cross validation

Interpretable approaches: prosodic, linguistic and cognitive features.

The whole experiment pipeline can be run through:

1) run_monolingual_experiments.sh for mono-lingual experiments; 
2) run_multilingual_experiments.sh for cross-lingual experiments; 
3) run_crosslingual_experiments.sh for cross-lingual experiments; 