# A Multi-modal Array of Interpretable Features to Evaluate Language and Speech Patterns in Different Neurological Disorders

In this project, we analyzed cognitive, linguistic, and acoustic patterns from the speech of different individuals with varying neurological disorders. 
The original recordings belong to two different data sets. The first, **NeuroLogical Signal (NLS)** is a private data set collected by
the authors of this study at Johns Hopkins Hospital and contains spoken responses to several tasks (i.e., reading passage,
object naming) from participants with varying NDs organized into categories and CN participants. Among the different NDs contained in the data
set, in this work we focused on three main types of NDs, namely, Parkinson's Disease (PD), Parkinson’s disease mimics (PDM), and Alzheimer's Disease.
To validate the results obtained for the  AD group, the same analysis was repeated using **ADReSSo
(ADR) 2021** challenge data set (Luz et al., 2020), obtained from the DementiaBank, including recordings from a CTP description
task. Only the training subset containing recordings and transcriptions from 87 AD participants and 79 CN participants
was adopted. This repository is a public open-source implementation that supports the extraction of cognitive, linguistic and acoustic features from speech recordings. This project has the aim of designing, extracting and analyzing speech and language production of subjects with different neurological disorders. 
In this repository we report the code that supports both the **feature extraction**, and the **statistical analysis** that we perform to evaluate the significance of the features between experimental groups.


## Research Article: 

In case you will use this code or take inspiration from it, plese cite out work: 
```

Favaro, A., Motley, C., Cao, T., Iglesias, M., Butala, A., Oh, E. S., Stevens, R. D., Villalba, J., Dehak, N., Moro-Velazquez, L. A Multi-Modal Array of Interpretable Features to Evaluate Language and Speech Patterns in Different Neurological Disorders. 2022 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2022

```
## Data Pre-processing:

* Recordings from both the data sets were resmapled to 16 kHz as required by the algorithms used for the feature extraction. To resample speech recordings to 16kHz run the script:
  * ```Data_Preprocessing/convert_to_16k.sh```.
* Spoken responses collected in the Cookie Theft task were automatically transcribed using a pre-trained conformer CTC  model (https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09) for the Librispeech data set built on top of icefall (https://github.com/k2-fsa/icefall). Transcriptions were manually supervised and corrected when needed,  
## Feature Extraction 

### Cognitive Features
For the extraction of part of the cognitive features we use a pre-trained conformer CTC model for the librispeech dataset built on top of icefall (https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09). 
To extract the cognitive features based on speech trascripts only, run the script```Cognitive/Extract_Cognitive_Features.py```

### Acoustic Features

* For the extraction of the acoustic features related to pause and speech time we use DigiPsych Prosody Repository (<https://github.com/NeuroLexDiagnostics/DigiPsych_Prosody>).
* For the extraction of the acoustic features related F0 and energy contour we use Disvoice Repository (<https://github.com/jcvasquezc/DisVoice/tree/master/prosody>).

To extract the acoustic features reported in the paper, run the script ```Acoustic/Extract_Acoustic_Features.py```



### Linguistic Features

* For the extraction of the linguistic features related to Part-of-Speech and Syntactic Complexity we use Spacy Python Library(<https://spacy.io/models>).
* For the extraction of the linguistic features related to Vocabulary Richness we use <https://pypi.org/project/lexicalrichness/>. 

To extract the linguistic features reported in the paper, run the script ```Linguistic/Extract_Linguistic_Features.sh```

## Statistical Analysis 

We report the main functions used to perform the statistical analysis to assess the significance of the features between experimental groups. 
In order to perform pair-wise Kruskal-Wallis H-tests, apply FDR correction, compute eta-squared effect-size and AUROC for each biomarker run the script ```Statistical_Analysis/utils.py```


## Reproducibility

Due to the licensing of the used data sets, we are not allowed to publish the recordings, the extracted features, nor the labels. However, we provided the source code to perform the feature extraction, and the statistical analysis reported in the paper. 


## Further references 

[1] Laureano Moro-Velazquez and Najim Dehak, “A review of the use of prosodic aspects of speech for the automatic detection and assessment of parkinson’s disease,” in Automatic Assessment of Parkinsonian Speech Workshop. Springer, 2019, pp. 42–59.”​

[2] Saturnino Luz, Fasih Haider, Sofia de la Fuente, Davida Fromm, and Brian MacWhinney, “Alzheimer’s dementia recognition through spontaneous speech: The address challenge,” arXiv preprint arXiv:2004.06833, 2020. ​

[3] Laureano Moro-Velazquez, Jorge A Gomez-Garcia, Juan Godino-Llorente, Francisco Grandas-Perez, Stefanie Shattuck-Hufnagel, Virginia Yagüe-Jimenez, and Najim Dehak, “Phonetic relevance and phonemic grouping of speech in the automatic detection of parkinson’s disease,” Scientific reports, vol. 9, no. 1, pp. 1–16, 2019​
​