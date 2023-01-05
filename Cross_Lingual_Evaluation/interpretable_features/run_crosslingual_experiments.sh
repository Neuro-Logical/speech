#!/usr/bin/env bash

# change to the full path of where the Non_Interpretable_features folder is
cd '/export/afavaro/git_code_version/speech/Cross_Lingual_Evaluation/interpretable_features/'

# To change the input/output directory used in each script, please change the input/output path located at the beginning of each Python script.


#  Mono-Lingual experiments
# Colombian

python classification/cross_lingual/colombian/AUROC/RP.py
python classification/cross_lingual/colombian/Models/RP.py
python classification/cross_lingual/colombian/SENS/RP.py

python classification/cross_lingual/colombian/AUROC/SS.py
python classification/cross_lingual/colombian/Models/SS.py
python classification/cross_lingual/colombian/SENS/SS.py

python classification/cross_lingual/colombian/AUROC/TDU.py
python classification/cross_lingual/colombian/Models/TDU.py
python classification/cross_lingual/colombian/SENS/SS.py

# Czech

python classification/cross_lingual/czech/AUROC/RP.py
python classification/cross_lingual/czech/Models/RP.py
python classification/cross_lingual/czech/SENS/RP.py

python classification/cross_lingual/czech/AUROC/SS.py
python classification/cross_lingual/czech/Models/SS.py
python classification/cross_lingual/czech/SENS/SS.py

# American English

python classification/cross_lingual/english/AUROC/RP.py
python classification/cross_lingual/english/Models/RP.py
python classification/cross_lingual/english/SENS/RP.py

python classification/cross_lingual/english/AUROC/SS.py
python classification/cross_lingual/english/Models/SS.py
python classification/cross_lingual/english/SENS/SS.py


# German

python classification/cross_lingual/german/AUROC/RP.py
python classification/cross_lingual/german/Models/RP.py
python classification/cross_lingual/german/SENS/RP.py

python classification/cross_lingual/german/AUROC/SS.py
python classification/cross_lingual/german/Models/SS.py
python classification/cross_lingual/german/SENS/SS.py

python classification/cross_lingual/german/AUROC/TDU.py
python classification/cross_lingual/german/Models/TDU.py
python classification/cross_lingual/german/SENS/TDU.py


# Italian

python classification/cross_lingual/italian/AUROC/RP.py
python classification/cross_lingual/italian/Models/RP.py
python classification/cross_lingual/italian/SENS/RP.py

python classification/cross_lingual/italian/AUROC/TDU.py
python classification/cross_lingual/italian/Models/TDU.py
python classification/cross_lingual/italian/SENS/TDU.py

# Castilian spanish

python classification/cross_lingual/spanish/AUROC/TDU.py
python classification/cross_lingual/spanish/Models/TDU.py
python classification/cross_lingual/spanish/SENS/SS.py

python classification/cross_lingual/spanish/AUROC/SS.py
python classification/cross_lingual/spanish/Models/SS.py
python classification/cross_lingual/spanish/SENS/SS.py