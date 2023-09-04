## Aim 

This repository is a public open-source implementation meant to support a multilingual evaluation of interpretable biomarkers to represent language and Speech Patterns in Parkinson’s Disease (PD).
This project has the aim of designing, extracting and analyzing speech and language production of subjects with Parkinson's Disease (PD). 
In this repository we report the code that supports both the **data pre-processing**, the **feature extraction**, the **statistical analysis**, and the **correlation analysis** that can be performed to evaluate both the effectiveness of the features in differentiating between CN and PD subjects as well as and their correlations with the strength of disorder using clinical scores (e.g., UPDRS).  This code is associated to the paper entitled (see reference in Acknowledgements):

``` 
 Multilingual Evaluation of Interpretable Biomarkers to Represent Language and Speech Patterns in Parkinson's Disease
```

## Set up  ⚙️ 

```
# Clone the repo
$ git clone https://github.com/Neuro-Logical/speech.git
$ cd Multilingual_Evaluation

# Create a virtual environment
$ python3 -m virtualenv .venv
$ source .venv/bin/activate

# Install the dependencies within the working environment with exact versions
$ pip install -r requirements.txt
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
##### Prepare data 
For the statistical and correlation analysis, the following data structure is required. For each family of features (i.e., acoustic, cognitive, linguistic), create a csv file structured as reported in the table below.

| Column name      | Data Type | Description  |
| :---        |    :----:   |          ---: |
| Feature name      | int or float  |Extracted feature value |
| Label   | string ("CN" or "PD")  | Classes needed to analyze statistical difference between experimental groups|
| Speaker ID | string or int | Unique identifier of each subject | 
| UPDRSIII  | float or int   | Unified Parkinson Disease Rating Scale (part 3) (value needed in the correlation analysis)|
| UPDRSIII-speech| float or int  | Unified Parkinson Disease Rating Scale (part 3) speech assessment (value needed in the correlation analysis)  |
| H&Y | float or int  | Hoehn & Yahr rating scale (value needed in the correlation analysis)|
| Task name | string (i.e., "CookieTheft") | Task from which the feature has ben extracted|

A mock csv file reproducing the structure above is reported in ``` data/data.csv ```.

#### Analysis

   - To perform pair-wise Kruskal-Wallis H-tests, apply FDR correction, compute eta-squared effect-size and AUROC for each biomarker use the functions in 

  ```
   compute_statistics/utils.py
   ```
   - To perform the correlation analysis between the feature and the clinical scores (UDRS III, UPDRS III part I (speech assessment), and Hoen and Yahr (H\&Y) scale), use the functions in  

  ``` 
    compute_statistics/compute_correlation.py
   ```
A practical application of the statistical analysis and of the correlation analysis is presented in the notebooks:
 ``` compute_statistics/compute_stats.ipynb```, and  ``` compute_statistics/correlation_analysis.ipynb```.


### Reproducibility

Due to the licensing of the used data sets, we are not allowed to publish the recordings, the features, nor the labels. However, we provide the source code to perform the feature extraction, the statistical, and the correlation analysis reported in the paper.
## Acknowledgements 🙏

In case you will use this code or take inspiration from it, please cite out work as follows:

``` 
@article{favaro2023multilingual,
  title={Multilingual evaluation of interpretable biomarkers to represent language and speech patterns in Parkinson's disease},
  author={Favaro, Anna and Moro-Vel{\'a}zquez, Laureano and Butala, Ankur and Motley, Chelsie and Cao, Tianyu and Stevens, Robert David and Villalba, Jes{\'u}s and Dehak, Najim},
  journal={Frontiers in Neurology},
  volume={14},
  pages={1142642},
  year={2023},
  publisher={Frontiers}
} 

```