# Quick start (fixed version)

```fish
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" python-multipart faster-whisper piper-tts torch librosa soundfile pyyaml levenshtein scipy matplotlib
```

**Web demo** - speak into the browser, get a transcript and a TTS voice back:

```fish
.venv/bin/python webapp/server.py   # then open http://localhost:8000
```

* *Transcribe + TTS* mode: Whisper (speech-to-text) + Piper (TTS). Works without training anything.
* *Parrotron* mode: direct speech-to-speech, enabled once `plz_load/best_parrotron*.pth` exists (or set `PARROTRON_CHECKPOINT`).
* Settings: `WHISPER_MODEL` (default `small.en`), `WHISPER_DEVICE` (`auto`/`cuda`/`cpu`), `PIPER_VOICE` (default `en_US-lessac-medium`).
* Whisper runs on the GPU only if the CUDA 12 cuBLAS is installed: `uv pip install --python .venv/bin/python nvidia-cublas-cu12`. Otherwise it uses the CPU.

**Training Parrotron**

```fish
# 1. build the dataset from LibriSpeech (https://www.openslr.org/12), TTS targets are made with Piper
.venv/bin/python make_dataset.py /path/LibriSpeech/train-clean-100 ./data/train ./label,csv/train.csv
.venv/bin/python make_dataset.py /path/LibriSpeech/test-clean ./data/test ./label,csv/test.csv
# 2. train (--no_asr for the model without the ASR decoder)
.venv/bin/python train.py --batch_size 8
# 3. convert a wav with the trained model
.venv/bin/python inference.py plz_load/best_parrotron.pth input.wav
```

---

# 아직 제작중입니다.

# Parrotron Speech-to-Speech
|Strategy|Dataset|loss|WER| 
|--------|-----|-------|------|
|E0|Librispeech_test-clean|?||*|
|E1|Librispeech_test-clean|?||?|

* E0 : Parrotron_no_ASR_decoder 
* E1 : Parrotron 

## Parrotron Intro
Pytorch implementation of "Parrotron: An End-to-End Speech-to-Speech Conversion Model and its Applications to Hearing-Impaired Speech and Speech Separation".

## Model
### Feature
* Input wav : 80-dim log-Mel

  parameter | value
  ------|-----
  N_FFT | 1024
  Frame length | 20ms
  Frame shift | 12.5ms
  Window function | hann window
  fmin | 125
  fmin | 7600

* Targets wav : 1025-dim STFT 

  parameter | value
  ------|-----
  FFT | 2048
  Frame length | 20ms
  Frame shift | 12.5ms
  Window function | hann window

### Architecture
<img width = "400" src = "https://user-images.githubusercontent.com/43025347/109615901-d7fdef00-7b77-11eb-8592-95c4ce285ce4.png">

### Print Model
```python

```

## Data
### Dataset information
Train data set : train-clean-100
Test data set : test-clean

LibriSpeech ASR corpus : https://www.openslr.org/12

### Data format
* 음성 데이터 : 16bit, mono 16k sampling WAV audio
* 정답 스크립트 : 
  ```js
    잠시보류
  ```

### Dataset folder structure
* DATASET-ROOT-FOLDER
```
|--DATA
   |--train
      |--wav
         +--a.wav, b.wav, c.wav ...
      |--TTS_wav
         +--a.wav, b.wav, c.wav ...
      |--txt
         +--a.txt, b.txt, c.txt ...
   |--test
      |--wav
         +--a_test.wav, b_test.wav, c_test.wav ...
      |--TTS_wav
         +--a_test.wav, b_test.wav, c_test.wav ...
      |--txt
         +--a_test.txt, b_test.txt, c_test.txt ...
```
* train.csv, test.csv
  ```
  <wav-path>,<script-path>,<TTS_wav-path>
  460-172359-0076.wav,460-172359-0076.txt,460-172359-0076.wav
  460-172359-0077.wav,460-172359-0077.txt,460-172359-0077.wav
  460-172359-0078.wav,460-172359-0078.txt,460-172359-0078.wav
  460-172359-0079.wav,460-172359-0079.txt,460-172359-0079.wav
  460-172359-0080.wav,460-172359-0080.txt,460-172359-0080.wav
  ...
  ```
* english_unit.labels
  ```
  #id\char 
  0   _
  1   A
  2   B
  3   C
  ...
  26   Z
  27   '
  28    
  29   <s>
  30   </s>
  ```

## References
### Git hub References
* https://github.com/NVIDIA/tacotron2
* https://github.com/sooftware

### Paper References
* Parrotron: An End-to-End Speech-to-Speech Conversion Model and its Applications to Hearing-Impaired Speech and Speech Separation (https://arxiv.org/abs/1904.04169)

## computer power
* NVIDIA TITAN Xp * 4

## Contacts
학부생의 귀여운 시도로 봐주시고 해당 작업에 대한 피드백, 문의사항 모두 환영합니다.

fd873630@naver.com로 메일주시면 최대한 빨리 답장드리겠습니다.