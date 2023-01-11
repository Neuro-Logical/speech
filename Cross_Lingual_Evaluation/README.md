### A Cross-lingual Evaluation of Interpretable and Non-interpretable Approaches for Automatic Detection of Parkinson's Disease from Language and Speech


This project aimed at comparing the performance of interpretable and non-interpretable biomarkers in the detection of Parkinson's Disease.

## Set up  ⚙️ 

```
# Clone the repo
$ git clone https://github.com/Neuro-Logical/speech.git
$ cd Cross_Lingual_Evaluation/

# Create a virtual environment
$ python3 -m virtualenv .venv
$ source .venv/bin/activate

# Install the dependencies within the working environment with exact versions
$ pip install -r requirements.txt
```
## Data pre-processing 📈

Recordings have to be resampled at 16 kHz, as required by the algorithms used for the feature extraction. To resample speech recordings to 16kHz, run the following command after having modified input and output paths respectively: 
  
  ```
  $ bash Data_prereprocessing/convert_to_16k.sh 
  ```

To extract the speech transcripts from the speech recordings using openAI Whisper (https://openai.com/blog/whisper), run the following command after having modified input and output paths respectively:

   ```
   $ python Data_preprocessing/extract_transcripts.py
   ```

## Interpretable and Non-interpretable Pipeline

Code for feature extraction and machine learning experiments using interpretable and non intepretable features is reported into  ```inter```