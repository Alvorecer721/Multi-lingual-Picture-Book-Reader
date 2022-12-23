import os
import queue
import sounddevice as sd
import vosk
import sys
import json

q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def detect_speech():
    samplerate = 48000
    model = vosk.Model("../vosk-en")

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=1, dtype='int16',
                           channels=1, callback=callback):
        print('#' * 80)
        print('Listening ...')
        print('#' * 80)

        rec = vosk.KaldiRecognizer(model, samplerate)

        speech = ""

        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                res = json.loads(rec.FinalResult())
                speech = speech + res['text'] + ", "
                print(res['text'])
                if res['text'] == "thank you":
                    return speech


if __name__ == "__main__":
    ans = detect_speech()
    print(ans)
