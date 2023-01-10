### A Cross-lingual Evaluation of Interpretable and Non-interpretable Approaches for Automatic Detection of Parkinson's Disease from Language and Speech

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
Recordings from both the data sets were resampled to 16 kHz as required by the algorithms used for the feature extraction. To resample speech recordings to 16kHz, run the script: 
  
  ```
  $ bash Data_prereprocessing/convert_to_16k.sh 
  ```

   - To extract speech transcripts from speech recordings using openAI Whisper (https://openai.com/blog/whisper), run:

   ```
      python Data_preprocessing/extract_transcripts.py
   ```
