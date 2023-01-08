#!/usr/bin/env bash

# change to the full path of where the interpretable_features folder is.
# Input and output paths are specified at the beginning of each script and should be be modified accordingly.

cd '/export/afavaro/git_code_version/speech/Cross_Lingual_Evaluation/interpretable_features/'

#  Mono-Lingual experiments
# Colombian Spanish

python classification/mono_lingual/colombian/AUROC/RP.py
python classification/mono_lingual/colombian/Models/RP.py
python classification/mono_lingual/colombian/SENS/RP.py

python classification/mono_lingual/colombian/AUROC/SS.py
python classification/mono_lingual/colombian/Models/SS.py
python classification/mono_lingual/colombian/SENS/SS.py

python classification/mono_lingual/colombian/AUROC/TDU.py
python classification/mono_lingual/colombian/Models/TDU.py
python classification/mono_lingual/colombian/SENS/SS.py

# Czech

python classification/mono_lingual/czech/AUROC/RP.py
python classification/mono_lingual/czech/Models/RP.py
python classification/mono_lingual/czech/SENS/RP.py

python classification/mono_lingual/czech/AUROC/SS.py
python classification/mono_lingual/czech/Models/SS.py
python classification/mono_lingual/czech/SENS/SS.py

# American english

python classification/mono_lingual/english/AUROC/RP.py
python classification/mono_lingual/english/Models/RP.py
python classification/mono_lingual/english/SENS/RP.py

python classification/mono_lingual/english/AUROC/SS.py
python classification/mono_lingual/english/Models/SS.py
python classification/mono_lingual/english/SENS/SS.py


# German

python classification/mono_lingual/german/AUROC/RP.py
python classification/mono_lingual/german/Models/RP.py
python classification/mono_lingual/german/SENS/RP.py

python classification/mono_lingual/german/AUROC/SS.py
python classification/mono_lingual/german/Models/SS.py
python classification/mono_lingual/german/SENS/SS.py

python classification/mono_lingual/german/AUROC/TDU.py
python classification/mono_lingual/german/Models/TDU.py
python classification/mono_lingual/german/SENS/TDU.py


# Italian

python classification/mono_lingual/italian/AUROC/RP.py
python classification/mono_lingual/italian/Models/RP.py
python classification/mono_lingual/italian/SENS/RP.py

python classification/mono_lingual/italian/AUROC/TDU.py
python classification/mono_lingual/italian/Models/TDU.py
python classification/mono_lingual/italian/SENS/TDU.py

# Castilian spanish

python classification/mono_lingual/spanish/AUROC/TDU.py
python classification/mono_lingual/spanish/Models/TDU.py
python classification/mono_lingual/spanish/SENS/SS.py

python classification/mono_lingual/spanish/AUROC/SS.py
python classification/mono_lingual/spanish/Models/SS.py
python classification/mono_lingual/spanish/SENS/SS.py