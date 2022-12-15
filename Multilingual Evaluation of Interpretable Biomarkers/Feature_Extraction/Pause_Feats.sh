KALDI_ROOT=/home/afavaro/kaldi
export PATH=$PATH:$KALDI_ROOT/src/featbin/



# Pause related features extracted with: https://github.com/NeuroLexDiagnostics/DigiPsych_Prosody

# f = frame length, namely the length of the frame used by the VAD adopted for the feature extraction.
# f can be 10, 20, 30 ms.
# -a is the path of folder containing the recordings from which

python featurize.py -a  /export/b15/afavaro/Frontiers/Czech_PD/All_16k/ -f 20 

python featurize.py -a  /export/b15/afavaro/Frontiers/German_PD/All/ -f 20 

python featurize.py -a  /export/b15/afavaro/Frontiers/GITA_PD_REAL/ -f 20 

python featurize.py -a  /export/b15/afavaro/Frontiers/Italian_PD/B1/ -f 20 

python featurize.py -a  /export/b15/afavaro/Frontiers/Italian_PD/B2/ -f 20 

python featurize.py -a  /export/b15/afavaro/Frontiers/German_PD/All/ -f 20 

python featurize.py -a  /export/b15/afavaro/Frontiers/GITA_PD_REAL/ -f 20 

python featurize.py -a /export/b15/afavaro/Frontiers/Neurovoz_data/Neurovoz_rec/ -f 20 

python featurize.py -a  /export/b15/afavaro/Frontiers/NLS/NLS_RESAMPLED/ -f 20 


