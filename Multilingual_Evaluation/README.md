## Aim 

This repository is a public open-source implementation meant to support a multilingual evaluation of interpretable biomarkers to represent language and Speech Patterns in Parkinson’s Disease (PD).
This project has the aim of designing, extracting and analyzing speech and language production of subjects with Parkinson's Disease (PD). 
In this repository we report the code that supports both the **data pre-processing**, the **feature extraction**, the **statistical analysis**, and the **correlation analysis** that can be performed to evaluate both the effectiveness of the features in differentiating between CN and PD subjects as well as and their correlations with the strength of disorder using clinical scores (e.g., UPDRS). This code is associated to the paper entitled:

``` 
 A MULTI-MODAL ARRAY OF INTERPRETABLE FEATURES TO EVALUATE LANGUAGE AND SPEECH PATTERNS IN DIFFERENT NEUROLOGICAL DISORDERS
```

## Experimental pipeline 💥
### Data pre-processing 📈

   - To resample the speech recordings to 16 kHz, run:

  ``` 
   $ bash data_preprocessing/convert_to_16k.sh
 
  ```
   - To extract speech transcripts from speech recordings using openAI Whisper (https://openai.com/blog/whisper), run:

   ```
      python data_preprocessing/get_speech_transcripts.py
   ```
### Feature extraction 🔨

   - To extract the ***cognitive*** features from the speech transcripts, see the script

   ``` 
      feature_extraction/extract_cognitive_features.py
  ```
   - To extract the ***linguistic*** features from speech transcripts, see the script

  ```
     feature_extraction/extract_linguistic_features.py
  ```
   - To extract the linguistic and cognitive features from the transcripts of your speech data set, you can use the notebook:

  ``` 
     feature_extraction/extract_ling_cog_features.ipynb
  ```
   - To extract the ***prosodic*** features from speech transcripts, change the input path (recording folder) and the output path (path to csv file containing the extracted features) and run the script 

  ```
    $ bash feature_extraction/extract_prosodic_features.sh 
  ```
   In the extraction of the prosodic features two libraries were used: ***Disvoice*** (https://github.com/jcvasquezc/DisVoice/tree/master/disvoice/prosody) and ***DigiPsych Prosody*** (https://github.com/NeuroLexDiagnostics/DigiPsych_Prosody).
#### Statistical and correlation analysis 📊

   - To perform pair-wise Kruskal-Wallis H-tests, apply FDR correction, compute eta-squared effect-size and AUROC for each biomarker use the functions in 

  ```
   compute_statistics/utils.py
   ```
   - To perform the correlation analysis between the feature and the clinical scores (UDRS III, UPDRS III part I (speech assessment), and Hoen and Yahr (H\&Y) scale), use the functions in  

  ``` 
    compute_statistics/compute_correlation.py
   ```
### Reproducibility

Due to the licensing of the used data sets, we are not allowed to publish the recordings, the features, nor the labels. However, we provide the source code to perform the feature extraction, the statistical, and the correlation analysis reported in the paper.
## Acknowledgements 🙏

In case you will use this code or take inspiration from it, please cite out work.