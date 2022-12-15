KALDI_ROOT=/home/afavaro/kaldi
export PATH=$PATH:$KALDI_ROOT/src/featbin/

#prosodic features extraction using Disvoice: https://github.com/jcvasquezc/DisVoice/tree/master/disvoice/prosody

python prosody.py "/export/b15/afavaro/Frontiers/Czech_PD/All_16k/" "/export/b15/afavaro/Frontiers/Czech_PD/prosody.csv" "true" "false" "csv"

python prosody.py  "/export/b15/afavaro/Frontiers/German_PD/All/" "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GERMAN/Acoustic/Prosody/prosody.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/GITA_PD_REAL/" "/export/b15/afavaro/Frontiers/GITA_PD/prosody.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/Italian_PD/B1/" "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/Acoustic/Prosody/B1_prosody.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/Italian_PD/B2/" "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/Acoustic/Prosody/B2_prosody.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/Neurovoz_data/Neurovoz_rec/" "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/Acoustic/Prosody/whole_prosody.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/NLS/NLS_RESAMPLED/" "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NLS/Acoustic/Prosody/prosody_last_meeting.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/Italian_PD/RPs_concatenated/" "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/Acoustic/Prosody/RP_concatenated.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/NLS/RPs_concatenated/" "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NLS/Acoustic/Prosody/RP_concatenated.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/GITA_NEW_TASKS/All_Recordings_Correct_Naming/" "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_concatenated.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/GITA_NEW_TASKS/Gita_Novel_resampled/" "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/Acoustic/Prosody/prosody.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/Italian_PD/FBR1/" "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/Acoustic/Prosody/FBR1_prosody.csv" "true" "false" "csv"