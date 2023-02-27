# Assignment

## Preparation

1. Create a virtual environment
```
venv .env
```
2. Activate the created virtual environment
```
./.env/Scripts/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Prompt #1 - Butterfly
Implemented in `find_butterfly.py`, this task searches butterflies in a frame of video, and locate them on the screen. It counts the number of butterflies in one frame and the change of number compared with the last frame.

To run the script, use:
```
python find_butterfly.py [--score_threshold SCORE_THRESHOLD] [--nms_threshold NMS_THRESHOLD] [--cuda] file
```
which:
- `file`: The butterfly video file to track. Required.
- `--score_threshold`: The score threshold for detected box to be taken account. Get harsh with large number.
- `--nms_threshold`: The IoU threshold used in Non Max Suppression. Detected box that overlapped with higher-scored box in a certain ratio is taken out from the result. Get harsh with small number.
- `--cuda`: Use cuda device whenever possible.

This uses `google/owlvit-base-patch32` as object detection backend. This model allows zero-shot object detection using text as the condition.

## Prompt #2 - Pa ta ka
This task is done in `count_ptk.py`. It reads an audio file, transcript it, and find how many substring `'ptk'` can be found.

To run the script, use:
```
python count_ptk.py file
```
where `file` is the path of audio file.

This use Wav2Vec2 as backend. In particular, the `facebook/wav2vec2-xlsr-53-phon-cv-ft` varient is used. Wav2Vec2 recognizes what the speaker has spoken. This varient specifically recognizes phonemes, the basic sound unit to distinguish one word. Then this script remove the 'a' phone and count 'ptk' substring.

Phoneme varient is chosen for language independence. For English scenario, `facebook/wav2vec2-base-960h` can also be used. In this case, the script should strip whitespace and count substring 'PATACA' instead of 'ptk'.