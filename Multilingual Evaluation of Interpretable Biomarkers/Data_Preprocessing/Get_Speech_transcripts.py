import os
import pandas as pd
import subprocess
pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 1000


def get_speech_transcripts(path_recordings, language):

    """Function to derive transcription from speech recordings using Whisper: https://openai.com/blog/whisper/.
    Path_recordings: path to the folder containing the recordings to transcribe.
    Language: the language of the speech recordings."""

    path_recordings_all = [os.path.join(path_recordings, base) for base in os.listdir(path_recordings)]
    for path in path_recordings_all:
        subprocess.run(f"whisper {path} --language {language}", shell=True)
