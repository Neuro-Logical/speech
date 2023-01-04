#!/usr/bin/env bash

# change to the full path of where the Non_Interpretable_features folder is
cd '/speech/Cross_Lingual_Evaluation/interpretable_features/'

#  Mono-Lingual experiments

# colombian

python grid_search/mono_lingual/colombian/RP.py
python grid_search/mono_lingual/colombian/SS.py
python grid_search/mono_lingual/colombian/TDU.py

# czech

python grid_search/mono_lingual/czech/SS.py
python grid_search/mono_lingual/czech/rp.py

# American english

python grid_search/mono_lingual/english/SS.py
python grid_search/mono_lingual/english/RP.py

# german

python grid_search/mono_lingual/german/SS.py
python grid_search/mono_lingual/german/RP.py
python grid_search/mono_lingual/german/TDU.py

# italian

python grid_search/mono_lingual/italian/RP.py
python grid_search/mono_lingual/italian/TDU.py

# Castilian spanish

python grid_search/mono_lingual/spanish/SS.py
python grid_search/mono_lingual/spanish/TDU.py


###################################################################

python grid_search/multi_Lingual/RP.py
python grid_search/multi_Lingual/TDU.py
python grid_search/multi_Lingual/SS.py

