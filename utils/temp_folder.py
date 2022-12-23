import os.path as path
import os
import shutil
import tempfile
from pdf2image import convert_from_path


class TempFolder(object):

    def __init__(self, prefix='temp_folder_'):
        self._path = tempfile.mkdtemp(prefix=prefix, dir=path.abspath(path.join(os.getcwd(), "..")))
        print('Create Temp Folder: {}'.format(self._path))

    def get_directory(self, sub_folder='.'):
        new_path = path.realpath(path.join(self._path, sub_folder))
        if path.normpath(new_path) == path.normpath('/'):
            raise Exception('Error, path is too dangerous.')
        return new_path

    def __del__(self):
        shutil.rmtree(self._path, True)
        print('Remove Temp Folder: {}'.format(self._path))


if __name__ == "__main__":
    # tf = TempFolder()
    # print(tf.get_directory())
    # input("Press Enter to continue...")

    pages = convert_from_path(r"D:\Individual_Project\individual_project\temp_folder_j1fydpbt\intelligentAgents4.pdf")
    for idx, page in enumerate(pages):
        page.save(r'D:\Individual_Project\individual_project\temp_folder_j1fydpbt\{}.jpg'.format(idx), 'JPEG')
