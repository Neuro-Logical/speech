#!/usr/bin/env bash

# change this to the full path of where the Non_Interpretable_features folder is
cd '/home/ytsai25/Non_Interpretable_features'

# set folder to save feature files (will have subfolders "xvector", "trill", "hubert", "wav2vec2")
feat_pth='/export/b11/ytsai25/feats/'
mkdir -p "${feat_pth}xvector"
mkdir -p "${feat_pth}trill"
mkdir -p "${feat_pth}hubert"
mkdir -p "${feat_pth}wav2vec2"

# set folder to save printed experiment results
out_pth='/home/ytsai25/Non_Interpretable_features/output/'
# input folder that contains all data sets
rec_pth='/export/b15/afavaro/Frontiers/'

# feature extraction (features already extracted and saved on the grid, uncomment this part if to re-extract)
# python feature_extraction/feature_extraction.py $rec_pth $feat_pth

# mono-lingual experiments ==================
# Colombian Spanish
python nested_cross_validation/monolingual/Colombian/TDU.py $feat_pth > "${out_pth}output_mono_gita_tdu.txt"
python nested_cross_validation/monolingual/Colombian/SS.py $feat_pth > "${out_pth}output_mono_gita_monologue.txt"
python nested_cross_validation/monolingual/Colombian/RP.py $feat_pth > "${out_pth}output_mono_gita_readpassage.txt"
# Czech
python nested_cross_validation/monolingual/Czech/RP.py $feat_pth > "${out_pth}output_mono_czech_readpassage.txt"
python nested_cross_validation/monolingual/Czech/SS.py $feat_pth > "${out_pth}output_mono_czech_monologue.txt"
# German
python nested_cross_validation/monolingual/German/TDU.py $feat_pth > "${out_pth}output_mono_ger_tdu.txt"
python nested_cross_validation/monolingual/German/SS.py $feat_pth > "${out_pth}output_mono_ger_monologue.txt"
python nested_cross_validation/monolingual/German/RP.py $feat_pth > "${out_pth}output_mono_ger_readpassage.txt"
# Italian
python nested_cross_validation/monolingual/Italian/TDU.py $feat_pth > "${out_pth}output_mono_ita_tdu.txt"
python nested_cross_validation/monolingual/Italian/RP.py $feat_pth > "${out_pth}output_mono_ita_readpassage.txt"
# English
python nested_cross_validation/monolingual/NLS/RP.py $feat_pth > "${out_pth}output_mono_nls_readpassage.txt"
python nested_cross_validation/monolingual/NLS/SS.py $feat_pth > "${out_pth}output_mono_nls_monologue.txt"
# Castilian spanish
python nested_cross_validation/monolingual/Spanish/TDU.py $feat_pth > "${out_pth}output_mono_neu_tdu.txt"
python nested_cross_validation/monolingual/Spanish/SS.py $feat_pth > "${out_pth}output_mono_neu_monologue.txt"

# multi-lingual experiments ==================
python nested_cross_validation/multilingual/TDU.py $feat_pth > "${out_pth}output_multi_tdu.txt"
python nested_cross_validation/multilingual/SS.py $feat_pth > "${out_pth}output_multi_monologue.txt"
python nested_cross_validation/multilingual/RP.py $feat_pth > "${out_pth}output_multi_readpassage.txt"

# cross-lingual experiments ==================
python nested_cross_validation/crosslingual/TDU.py $feat_pth > "${out_pth}output_cross_tdu.txt"
python nested_cross_validation/crosslingual/SS.py $feat_pth > "${out_pth}output_cross_monologue.txt"
python nested_cross_validation/crosslingual/RP.py $feat_pth > "${out_pth}output_cross_readpassage.txt"





