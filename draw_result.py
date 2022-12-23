# %%
import matplotlib.pyplot as plt
import numpy as np
from utils.text import detect_language

languages = ['chi_sim', 'eng', 'fra', 'jpn', 'ara']
images = ['image/test_Chinese.png',
          'image/test_English.png',
          'image/test_French.png',
          'image/test_Japanese.png',
          'image/test_Arabic.png']


def plot_language_confidence(lan):
    plt.style.use('seaborn')
    lineStyles = ['-', '--', '-.', ':']
    _, wordConf, _ = detect_language(images[languages.index(lan)], languages)

    for idx, line in enumerate(wordConf):
        numLineStyle = len(lineStyles)
        plt.plot(line, linestyle=lineStyles[idx % numLineStyle], color="C{}".format(idx), label=languages[idx])

    plt.legend()
    plt.title("Compare Individual Word Confidence for Story Book Image in {}".format(lan))
    plt.show()


def bar_image_score():
    plt.style.use('seaborn')
    bestScoreList = []
    for idx, img in enumerate(images):
        _, _, bestScore = detect_language(img, languages)
        bestScoreList.append(bestScore)

    bestScoreList_transpose = np.transpose(bestScoreList)

    plt.subplots()
    index = np.arange(len(images))
    bar_width = 0.15
    opacity = 1

    for idx, img in enumerate(images):
        plt.bar(index + idx * bar_width, bestScoreList_transpose[idx].tolist(), bar_width,
                alpha=opacity,
                color='C{}'.format(idx + 1),
                label=languages[idx])

    plt.ylabel('Scores')
    plt.title('Scores of Language by Image')
    plt.xticks(index + bar_width, list(map(lambda x: x.lstrip("image/"), images)))
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    for language in languages:
        plot_language_confidence(language)
    bar_image_score()

# %%

from PIL import Image
import sys
from tesserocr import PyTessBaseAPI
import cv2
import pytesseract

testI = "D:/Individual_Project/individual_project/5.jpg"

# Grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread(testI)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]

options = "-l {}".format('jpn')
text = pytesseract.image_to_string(blur, config=options)
print(text)