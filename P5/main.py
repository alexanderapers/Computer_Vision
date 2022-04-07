import cv2
import files

def main():
    # gathers the two datasets file names and class labels
    # they are already split into train, validation and test using stratification
    # train, validation and test variables are tuples of lists (file_names, label_names)
    # classes are the unique names of the classes that are used
    stanford_train, stanford_validation, stanford_test, stanford_classes = files.process_stanford40()
    TVHI_train, TVHI_validation, TVHI_test, TVHI_classes = files.process_TVHI()

if __name__ == "__main__":
    main()
