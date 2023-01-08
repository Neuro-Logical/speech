import os
import whisper


def extract_speech_transcripts(path_to_recordings, output_folder):

    """ Function that extract speech transcripts using OpenAI's Whisper.
    path_to_recordings: path to the folder containing the recordings to transcribe.
    output_folder: path to the folder where the transcripts of each recording will be stored.
    This function outputs a text file for each recording with the corresponding transcriptions. """

    paths = [os.path.join(path_to_recordings, elem) for elem in os.listdir(path_to_recordings)]

    # keep only non-empty recordings (> 56 Bytes)
    files = []
    for m in paths:
        size = os.stat(m).st_size / 1000
        if size > 56:
            files.append(m)

    # extract and save transcripts.
    for i in files:
        # print(i)
        model = whisper.load_model("medium")
        result = model.transcribe(i)
        test = result['text']
        base = os.path.basename(i).split(".wav")[0]
        total = os.path.join(output_folder, base + ".txt")
        text_file = open(total, "wt")
        n = text_file.write(test)
        text_file.close()
