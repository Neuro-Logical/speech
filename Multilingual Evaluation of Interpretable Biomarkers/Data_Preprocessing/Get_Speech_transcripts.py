import os
import pandas as pd
import subprocess
pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 1000


def get_speech_transcripts(path_recordings, language):
    path_recordings_all = [os.path.join(path_recordings, base) for base in os.listdir(path_recordings)]
    for path in path_recordings_all:
        subprocess.run(f"whisper {path} --language {language}", shell=True)
