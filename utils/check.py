import os


def file_existence(path):
    """
    check if the file exists on the local drive or not
    :param path: the path to the file
    :return: Boolean
    """
    return os.path.isfile(path)