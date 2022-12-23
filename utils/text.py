import cv2
import pytesseract
from pytesseract import Output
from utils.speech import generate_speech
from statistics import mean
from collections import deque
from statistics import pvariance

lang_code = {
    'chi_sim': 'zh-cn',
    'eng': 'en',
    'fra': 'fr',
    'jpn': 'ja'
    # Tesseract does not work properly with arabic in Windows
    # 'ara': 'ar'
}


def count_greater(d):
    """
    count how many elements in the target list is the greatest among all lists
    :param d: dequeued list
    :return: count
    """
    numElement = len(d)
    minLength = len(min(d, key=len))
    count = 0

    for i in range(minLength):
        if all_greater(i, numElement, d):
            count += 1

    return count


def all_greater(n, m, d):
    """
    check whether the specified element is the greatest
    :param n: number of lists in the dequeued list
    :param m: length of shortest list in the dequeued list
    :param d: dequeued list
    :return: true or false
    """
    flag = True
    for i in range(1, m):
        if d[0][n] < d[i][n]:
            flag = False

    return flag


def get_count_list(listOfList):
    """
    rotate the list in the list of list to get count for each list
    :param listOfList: data takes in
    :return: a list with each count for each list in the list of list
    """
    d = deque(listOfList)
    numElement = len(d)

    counts = []

    for c in range(numElement):
        counts.append(count_greater(d))
        d.rotate(-1)
    return counts


def thresh_image(imgPath):
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]

    return thresh


def word_confidence_list(lan, img):
    """
    calculate each individual word confidence, as well as the average confidence of the whole butch of words
    :param img: file path for the image
    :param lan: language select to be identified
    :return: a list containing confidence for each word
    """
    options = "-l {}".format(lan)
    conf = pytesseract.image_to_data(thresh_image(img), config=options, output_type=Output.DICT)['conf']
    while '-1' in conf:
        conf.remove('-1')
    return conf


def detect_language(img, languestList):
    """
    detect the language in the image with the best score
    :param languestList: a list of language code for each testing language
    :param img: file path for the image
    :return: detected language code as string, list of
    """
    word_confidence = []
    overall_confidence = []

    for idx, lang in enumerate(languestList):
        word_confidence.append(word_confidence_list(lang, img))
        overall_confidence.append(mean(word_confidence_list(lang, img)))
        print("{} confidence is : {}, overall confidence is {}".format(lang, word_confidence[idx],
                                                                       overall_confidence[idx]))

    score = get_count_list(word_confidence)
    bestIndex = overall_confidence.index(max(overall_confidence))
    bestLang = languestList[bestIndex]

    print(score)
    print("The language detected is {}, with a confidence of {} and a score of {}".format(bestLang, mean(
        word_confidence_list(bestLang, img)), score[bestIndex]))

    return bestLang, overall_confidence, score


def extract_text(file_path, lang):
    """
    extract text from image according to the language using cv2
    Grayscale, Gaussian blur, Otsu's threshold

    :param file_path: path for the input image
    :param lang: languages of the text
    :return: extracted text from the image
    """
    options = "-l {}".format(lang)
    text = pytesseract.image_to_string(thresh_image(file_path), config=options)
    return text


def identify_extract_read(target_image):
    """
    Given the image, first identify the language of the text in the image,
    then extract the text into machine-readable strings, then generate speech based on the extracted text
    :param target_image: path of the image
    :return:
    """
    available_languages = list(lang_code.keys())
    lang, _, _ = detect_language(target_image, available_languages)
    text = extract_text(target_image, lang)
    generate_speech(text, lang_code[lang])


if __name__ == "__main__":
    # Tesseract does not work properly with arabic in Windows
    # languages = ['chi_sim', 'eng', 'fra', 'jpn', 'ara']
    languages = ['chi_sim', 'eng', 'fra', 'jpn']
    # image = "../temp_folder_yhgvje33/0.jpg"

    testI = "../temp_folder_znizencl/0.jpg"


