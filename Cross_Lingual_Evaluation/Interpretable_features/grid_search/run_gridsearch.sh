#!/usr/bin/env bash

# change to the full path of where the Interpretable_features folder is
cd '/export/afavaro/git_code_version/speech/Cross_Lingual_Evaluation/Interpretable_features/'

#  Mono-Lingual experiments

# Colombian

python grid_search/mono_lingual/colombian/RP.py
python grid_search/mono_lingual/colombian/SS.py
python grid_search/mono_lingual/colombian/TDU.py

# Czech

python grid_search/mono_lingual/czech/SS.py
python grid_search/mono_lingual/czech/rp.py

# American english
python grid_search/mono_lingual/english/SS.py
python grid_search/mono_lingual/english/RP.py

# german

python grid_search/mono_lingual/german/SS.py
python grid_search/mono_lingual/german/RP.py
python grid_search/mono_lingual/german/TDU.py

# Italian

python grid_search/mono_lingual/italian/RP.py
python grid_search/mono_lingual/italian/TDU.py

# Castilian Spanish
python grid_search/mono_lingual/spanish/SS.py
python grid_search/mono_lingual/spanish/TDU.py

###################################################################

# Multi-Lingual experiments
# unique configurations of hyperparameters for all the languages, since the training set is invariant across experiments.
# Experiments are performed task wise.

python grid_search/multi_Lingual/RP.py
python grid_search/multi_Lingual/TDU.py
python grid_search/multi_Lingual/SS.py

###################################################################

# Cross-Lingual experiments.
# The same hyperparameters used in the multi-lingual experiments are used in the cross-lingual experiments.

