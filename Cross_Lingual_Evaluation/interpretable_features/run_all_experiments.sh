#!/usr/bin/env bash

# change to the full path of where the Non_Interpretable_features folder is
cd '/speech/Cross_Lingual_Evaluation/interpretable_features/'

#  Mono-Lingual experiments

# colombian

python nested_cross_validation/mono_lingual/colombian/AUROC/RP.py
python nested_cross_validation/mono_lingual/colombian/Models/RP.py
python nested_cross_validation/mono_lingual/colombian/SENS/RP.py

python nested_cross_validation/mono_lingual/colombian/AUROC/SS.py
python nested_cross_validation/mono_lingual/colombian/Models/SS.py
python nested_cross_validation/mono_lingual/colombian/SENS/SS.py

python nested_cross_validation/mono_lingual/colombian/AUROC/TDU.py
python nested_cross_validation/mono_lingual/colombian/Models/TDU.py
python nested_cross_validation/mono_lingual/colombian/SENS/SS.py
