# Multilingual Evaluation of Interpretable Biomarkers to Represent Language and Speech Patterns in Parkinson’s Disease

## Research aim 

Even though motor speech problems represent an early sign of Parkinson's Disease (PD), non-motor symptoms such as cognitive and linguistic impairments are also prevalent. Interpretable biomarkers derived from speech can help clinicians perform PD diagnosis and monitor the disorder's evolution over time. This work focuses on the multilingual evaluation of a composite array of biomarkers that can assist PD evaluation from speech. Most of previous works on automatic PD detection mainly analyzed acoustic biomarkers connected to hypokinetic dysarthria, a motor speech disorder associated with PD, and considered only a few languages and tasks simultaneously. In this work, we explored the acoustic, linguistic, and cognitive information encoded in the speech of several cohorts with PD subjects. Twenty-three biomarkers were analyzed from American English, Italian, Castilian Spanish, Colombian Spanish, German, and Czech by conducting a statistical analysis to evaluate which biomarkers better differentiate PD from healthy participants. The study leverages the concept of language robustness as a criterion in which a biomarker behaves the same, independently of the language. Hence, we propose a set of speech-based biomarkers that can effectively help evaluating PD while being language-independent. Biomarkers defining monopitch, pause time, pause percentage, silence duration, and speech rhythm provided better discriminability between experimental groups across languages. Similar conclusions were obtained for the linguistic biomarkers representing the length of the narratives and the syntactic categories of nouns and auxiliaries. Altogether, besides being significant, these biomarkers satisfied the language robustness requirements. As such, they can be adopted as interpretable biomarkers in the clinical practice across languages.

## Experimental pipeline 

The experimental pipeline followed in our paper goes as follows:

### 1) Data pre-processing

   - To resample the speech recordings to 16 kHz, run the script ```data_preprocessing/convert_to_16k.sh```

   - To extract speech transcripts from speech recordings using Whisper (https://openai.com/blog/whisper), run ```data_preprocessing/get_speech_transcripts.py```

### 3) Feature extraction:

   - To extract the ***cognitive*** features from the speech transcripts, run the script named ```feature_extraction/extract_cognitive_features.py```
   
   - To extract the ***linguistic*** features from speech transcripts, run the script named ```feature_extraction/extract_linguistic_features.py```
   
   - To extract the ***prosodic*** features from speech transcripts run the script named  ```feature_extraction/extract_prosodic_features```. In the extraction of the prosodic features two 
   libraries were used: Disvoice (https://github.com/jcvasquezc/DisVoice/tree/master/disvoice/prosody) and DigiPsych Prosody (https://github.com/NeuroLexDiagnostics/DigiPsych_Prosody)

#### 4) Statistical and correlation analysis

   - To perform pair-wise Kruskal-Wallis H-tests, apply FDR correction, compute eta-squared effect-size and AUROC for each biomarker use the functions in ```compute_statistics/utils.py```
   
   - To perform the correlation analysis between the feature and the clinical scores (UDRS III, UPDRS III part I (speech assessment), and Hoen and Yahr (H\&Y) scale), use the functions in  ```compute_statistics/compute_correlation.py```
   
### Reproducibility

Due to the licensing of the used data sets, we are not allowed to publish the recordings, the features, nor the labels. However, we provide the source code to perform the feature extraction, the statistical, and the correlation analysis reported in the paper. 


