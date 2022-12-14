


Code associated to the paper: 

Favaro, A., Motley, C., Cao, T., Iglesias, M., Butala, A., Oh, E. S., Stevens, R. D., Villalba, J., Dehak, N., Moro-
Velazquez, L. A Multi-Modal Array of Interpretable Features to Evaluate Language and Speech Patterns in Different
Neurological Disorders in The 2022 IEEE Spoken Language Technology Workshop (SLT 2022) 

This repository is a public open-source implementation that supports the extraction of cognitive, linguistic and acoustic features from speech recordings. This project has the aim of designing, extracting and analyzing speech and language production of subjects with different neurological disorders. 
In this repository we report the code that supports both the **feature extraction** and the the **statistical analysis** that we perform to evaluate the significance of the features between experimental groups.
 
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
