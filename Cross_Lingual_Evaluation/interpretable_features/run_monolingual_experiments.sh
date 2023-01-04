#!/usr/bin/env bash

# change to the full path of where the Non_Interpretable_features folder is
cd '/speech/Cross_Lingual_Evaluation/interpretable_features/'

#  Mono-Lingual experiments

# Colombian

python nested_cross_validation/mono_lingual/colombian/AUROC/RP.py
python nested_cross_validation/mono_lingual/colombian/Models/RP.py
python nested_cross_validation/mono_lingual/colombian/SENS/RP.py

python nested_cross_validation/mono_lingual/colombian/AUROC/SS.py
python nested_cross_validation/mono_lingual/colombian/Models/SS.py
python nested_cross_validation/mono_lingual/colombian/SENS/SS.py

python nested_cross_validation/mono_lingual/colombian/AUROC/TDU.py
python nested_cross_validation/mono_lingual/colombian/Models/TDU.py
python nested_cross_validation/mono_lingual/colombian/SENS/SS.py

# Czech

python nested_cross_validation/mono_lingual/czech/AUROC/RP.py
python nested_cross_validation/mono_lingual/czech/Models/RP.py
python nested_cross_validation/mono_lingual/czech/SENS/RP.py

python nested_cross_validation/mono_lingual/czech/AUROC/SS.py
python nested_cross_validation/mono_lingual/czech/Models/SS.py
python nested_cross_validation/mono_lingual/czech/SENS/SS.py

# American English

python nested_cross_validation/mono_lingual/english/AUROC/RP.py
python nested_cross_validation/mono_lingual/english/Models/RP.py
python nested_cross_validation/mono_lingual/english/SENS/RP.py

python nested_cross_validation/mono_lingual/english/AUROC/SS.py
python nested_cross_validation/mono_lingual/english/Models/SS.py
python nested_cross_validation/mono_lingual/english/SENS/SS.py


# German

python nested_cross_validation/mono_lingual/german/AUROC/RP.py
python nested_cross_validation/mono_lingual/german/Models/RP.py
python nested_cross_validation/mono_lingual/german/SENS/RP.py

python nested_cross_validation/mono_lingual/german/AUROC/SS.py
python nested_cross_validation/mono_lingual/german/Models/SS.py
python nested_cross_validation/mono_lingual/german/SENS/SS.py

python nested_cross_validation/mono_lingual/german/AUROC/TDU.py
python nested_cross_validation/mono_lingual/german/Models/TDU.py
python nested_cross_validation/mono_lingual/german/SENS/TDU.py


# Italian

python nested_cross_validation/mono_lingual/italian/AUROC/RP.py
python nested_cross_validation/mono_lingual/italian/Models/RP.py
python nested_cross_validation/mono_lingual/italian/SENS/RP.py

python nested_cross_validation/mono_lingual/italian/AUROC/TDU.py
python nested_cross_validation/mono_lingual/italian/Models/TDU.py
python nested_cross_validation/mono_lingual/italian/SENS/TDU.py

# Castilian Spanish

python nested_cross_validation/mono_lingual/spanish/AUROC/TDU.py
python nested_cross_validation/mono_lingual/spanish/Models/TDU.py
python nested_cross_validation/mono_lingual/spanish/SENS/SS.py

python nested_cross_validation/mono_lingual/spanish/AUROC/SS.py
python nested_cross_validation/mono_lingual/spanish/Models/SS.py
python nested_cross_validation/mono_lingual/spanish/SENS/SS.py