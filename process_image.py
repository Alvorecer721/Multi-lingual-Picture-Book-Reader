# %%
from PIL import Image
import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt


# %%
def get_text_location(img_path, language):
    img2dict = pytesseract.image_to_data(Image.open(img_path), lang=language, output_type=Output.DATAFRAME)
    locations = []
    num_boxes = len(img2dict)

    for i in range(num_boxes):
        if type(img2dict['text'][i]) != float:
            if " " not in img2dict['text'][i]:
                loc = [img2dict['left'][i], img2dict['top'][i], img2dict['width'][i], img2dict['height'][i]]
                locations.append(loc)

    return locations


def add_box_to_text(img_path, language):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    locations = get_text_location(img_path, language)

    for loc in locations:
        img = cv2.rectangle(img, (loc[0], loc[1]), (loc[0] + loc[2], loc[1] + loc[3]), (255, 0, 0), 4)

    _, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')

    plt.show()


# %%

if __name__ == '__main__':
    # add_box_to_text(r"D:\Individual_Project\individual_project\ostu.jpg", "jpn")
    # add_box_to_text(r"D:\Individual_Project\individual_project\127.jpg", "jpn")
    # add_box_to_text(r"D:\Individual_Project\individual_project\AGT.jpg", "jpn")
    # add_box_to_text(r"D:\Individual_Project\individual_project\AMT.jpg", "jpn")
    # add_box_to_text(r"D:\Individual_Project\individual_project\AMT.jpg", "jpn")
    add_box_to_text(r"D:\Individual_Project\individual_project\utils\5.jpg", "jpn")

# %%
