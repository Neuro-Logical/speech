# A Multi-modal Array of Interpretable Features to Evaluate Language and Speech Patterns in Different Neurological Disorders

In this project, we analyzed cognitive, linguistic, and acoustic patterns from the speech of different individuals with varying neurological disorders. 
The original recordings belong to two different data sets. The first, **NeuroLogical Signal (NLS)** is a private data set collected by
the authors of this study at Johns Hopkins Hospital and contains spoken responses to several tasks (i.e., reading passage,
object naming) from participants with varying NDs organized into categories and CN participants. Among the different NDs contained in the data
set, in this work we focused on three main types of NDs, namely, Parkinson's Disease (PD), Parkinson’s disease mimics (PDM), and Alzheimer's Disease.
To validate the results obtained for the  AD group, the same analysis was repeated using **ADReSSo
(ADR) 2021** challenge data set (Luz et al., 2020), obtained from the DementiaBank, including recordings from a CTP description
task. Only the training subset containing recordings and transcriptions from 87 AD participants and 79 CN participants
was adopted.

Detailed description of the ADR data set can be found at: Saturnino Luz, Fasih Haider, Sofia de la Fuente, Davida
Fromm, and Brian MacWhinney, “Alzheimer’s dementia
recognition through spontaneous speech: The adress
challenge,” arXiv preprint arXiv:2004.06833, 2020.

Favaro, A., Motley, C., Cao, T., Iglesias, M., Butala, A., Oh, E. S., Stevens, R. D., Villalba, J., Dehak, N., Moro-
Velazquez, L. A Multi-Modal Array of Interpretable Features to Evaluate Language and Speech Patterns in Different
Neurological Disorders in The 2022 IEEE Spoken Language Technology Workshop (SLT 2023) 

This repository is a public open-source implementation that supports the extraction of cognitive, linguistic and acoustic features from speech recordings. This project has the aim of designing, extracting and analyzing speech and language production of subjects with different neurological disorders. 
In this repository we report the code that supports both the **feature extraction**, and the **statistical analysis** that we perform to evaluate the significance of the features between experimental groups.
 
## Feature Extraction 

### Cognitive Features
For the extraction of part of the the cognitive features we use a pre-trained conformer CTC model for the librispeech dataset built on top of icefall (https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09). 

### Acoustic Features
* For the extraction of the acoustic features related to pause and speech time we use DigiPsych Prosody Repository (<https://github.com/NeuroLexDiagnostics/DigiPsych_Prosody>).
* For the extraction of the acoustic features related F0 and energy contour we use Disvoice Repository (<https://github.com/jcvasquezc/DisVoice/tree/master/prosody>).


### Linguistic Features
* For the extraction of the linguistic features related to Part-of-Speech and Syntactic Complexity we use Spacy Python Library(<https://spacy.io/models>).
* For the extraction of the linguistic features related to Vocabulary Richness we use <https://pypi.org/project/lexicalrichness/>. 

## Statistical Analysis 

We report an example of the statistical analysis that we perform in order to asses the significance of the features between experimental groups. 


### Reproducibility

Due to the licensing of the used data sets, we are not allowed to publish the recordings, the extracted features, nor the labels. However, we provided the source code to perform the feature extraction, and the statistical reported in the paper. 


#### References 

Saturnino Luz, Fasih Haider, Sofia de la Fuente, Davida Fromm, and Brian MacWhinney, “Alzheimer’s dementia
recognition through spontaneous speech: The adress challenge,” arXiv preprint arXiv:2004.06833, 2020.