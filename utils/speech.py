from gtts import gTTS
from os import system, remove
from playsound import playsound
from sys import platform


def generate_speech(text, lang='en'):
    audio_created = gTTS(text=text, lang=lang, slow=False)
    audio_created.save("read.mp3")

    if platform == "darwin":  # OS X
        system(f'afplay read.mp3')
    elif platform == "win32":  # Windows
        playsound("read.mp3")

    remove("read.mp3")


if __name__ == '__main__':
    text_to_read = "hello"
    # language = 'en'

    generate_speech(text_to_read)
