# A Multi-modal Array of Interpretable Features

## What is it 🔎
This repository is a public open-source implementation that supports the extraction of cognitive, linguistic and acoustic features from speech recordings. This project has the aim of designing, extracting and analyzing speech and language production of subjects with different neurological disorders. 
In this repository we report the code that supports both the  **data pre-processing**, **feature extraction**, and the **statistical analysis** that we perform to evaluate the significance of the features between experimental groups.

## Data pre-processing 📈
* Recordings from both the data sets were resmapled to 16 kHz as required by the algorithms used for the feature extraction. To resample speech recordings to 16kHz, run the script: 
  
  ```
  $ bash Data_Preprocessing/convert_to_16k.sh 
  ```
* Spoken responses were automatically transcribed using a pre-trained conformer CTC  model (https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09) for the Librispeech data set built on top of icefall (https://github.com/k2-fsa/icefall). Transcriptions were manually supervised and corrected when needed. For each recording a transcription is generated in capital letters without punctuation marks. The steps followed to get speech transcripts and alignment are:

  1) Prepare your data. Please see https://lhotse.readthedocs.io/en/latest/corpus.html#adding-new-corpora for more information. You can find various recipes for different datasets in https://github.com/lhotse-speech/lhotse/tree/master/lhotse/recipes.
  2) Follow https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/prepare.sh to extract features for your dataset.
  3) Adapt https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/tdnn_lstm_ctc/asr_datamodule.py to your dataset.
  4) Train a model for your dataset. Please see https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/train.py. For this step, we did not train but use a pretrained English model.
  5) Get alignments. Please see https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/ali.py

  To play with Google Colab using your own audios, you can use the notebook in:

  ```
  Data_Preprocessing/generate_transcripts.ipynb 
  ```
## Feature extraction 🔨
### Cognitive features 🧠
* To extract the cognitive features based on speech transcripts only, the code can be found in:

  ``` 
  Feature_Extraction/Cognitive/extract_cognitive_features.py
  ```
### Linguistic Features 🔡
* For the extraction of the linguistic features related to Part-of-Speech and Syntactic Complexity we use Spacy Python Library(<https://spacy.io/models>).
* For the extraction of the linguistic features related to Vocabulary Richness we use <https://pypi.org/project/lexicalrichness/>. 

To extract the linguistic features reported in the paper, run:

  ``` 
  python Feature_Extraction/Linguistic/extract_linguistic_features.py
  ```
To extract the linguistic and cognitive features from the transcripts of your speech data set, you can use the notebook in:

 ``` 
  Feature_Extraction/extract_ling_cog_features.ipynb
  ```
### Acoustic features 🔊
For the extraction of the acoustic features related to pause and speech time we use DigiPsych Prosody Repository (<https://github.com/NeuroLexDiagnostics/DigiPsych_Prosody>). For the extraction of the acoustic features related F0 and energy contour we use Disvoice Repository (<https://github.com/jcvasquezc/DisVoice/tree/master/prosody>).
To extract the acoustic features reported in the paper, run:

  ``` 
  $ bash Feature_Extraction/Acoustic/extract_acoustic_features.sh
  ```
## Statistical analysis 📊
In order to perform pair-wise Kruskal-Wallis H-tests, apply FDR correction, compute eta-squared effect-size and AUROC for each biomarker, see the functions in the script:

  ``` 
  Statistical_Analysis/utils.py
  ```
## Reproducibility
Due to the licensing of the used data sets, we are not allowed to publish the recordings, the extracted features, nor the labels. However, we provided the source code to perform the feature extraction, and the statistical analysis reported in the paper.

## Acknowledgements 🙏
In case you will use this code or take inspiration from it, please cite out work: 
```

Favaro, A., Motley, C., Cao, T., Iglesias, M., Butala, A., Oh, E. S., Stevens, R. D., Villalba, J., Dehak, N., Moro-Velazquez, L. A Multi-Modal Array of Interpretable Features to Evaluate Language and Speech Patterns in Different Neurological Disorders. 2022 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2022

```
## Further references 📖
[1] Laureano Moro-Velazquez and Najim Dehak, “A review of the use of prosodic aspects of speech for the automatic detection and assessment of parkinson’s disease,” in Automatic Assessment of Parkinsonian Speech Workshop. Springer, 2019, pp. 42–59.”​

[2] Saturnino Luz, Fasih Haider, Sofia de la Fuente, Davida Fromm, and Brian MacWhinney, “Alzheimer’s dementia recognition through spontaneous speech: The address challenge,” arXiv preprint arXiv:2004.06833, 2020. ​

[3] Laureano Moro-Velazquez, Jorge A Gomez-Garcia, Juan Godino-Llorente, Francisco Grandas-Perez, Stefanie Shattuck-Hufnagel, Virginia Yagüe-Jimenez, and Najim Dehak, “Phonetic relevance and phonemic grouping of speech in the automatic detection of parkinson’s disease,” Scientific reports, vol. 9, no. 1, pp. 1–16, 2019​
​