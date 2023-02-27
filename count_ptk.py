import argparse
import torch
import torchaudio
import torchaudio.functional as AF
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from typing import List

vowel_list = ['ɑ', 'ɒ']

def remove_substrs(src: str, tgt: List[str]):
    tgt_sorted = sorted(tgt, key=len, reverse=True)
    for substring in tgt_sorted:
        src.replace(substring, '')
    return src

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Audio file')
    args = parser.parse_args()
    filepath = args.file
    if not os.path.isfile(filepath):
        print(f'file {filepath} not found')
        exit(1)
        
    torch.set_grad_enabled(False)
    print('Loading pretrained models')
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-phon-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-phon-cv-ft")
    
    print('Loading audio file')
    y, sr = torchaudio.load(filepath, channels_first=True)
    y = y.mean(dim=0)
    # this pretrained model only accepts audio with 16000 sampling rate.
    # resample it before inputting to the pipeline
    print('Processing')
    yr = AF.resample(y, sr, 16000)
    extracted_feat = processor(yr, sampling_rate=16000, return_tensors='pt')
    logits = model(**extracted_feat).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # transcribe speech
    transcription = processor.batch_decode(predicted_ids)[0]
    print(f'Find phones: {transcription}')
    pruned = remove_substrs(transcription, vowel_list)
    print(f'ptk count: {pruned.count("ptk")}')