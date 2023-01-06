import os
import pandas as pd
import subprocess
pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 1000


def extract_transcripts(path_recordings, language):

    """ Function to extract transcription from speech recordings using Whisper: https://openai.com/blog/whisper/.
    Path_recordings: path to the folder containing the recordings to transcribe.
    Language: the language of the speech recordings.
    This function outputs a transcription for each recording, with the same name of the original file but with a different extension (.txt)."""

    path_recordings_all = [os.path.join(path_recordings, base) for base in os.listdir(path_recordings)]
    for path in path_recordings_all:
        subprocess.run(f"whisper {path} --language {language}", shell=True)
