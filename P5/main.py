import files
import load_data

def main():
    # gathers the two datasets file names and class labels
    # they are already split into train, validation and test using stratification
    # train, validation and test variables are tuples of lists (file_names, label_names)
    # classes are the unique names of the classes that are used
    stanford_train, stanford_validation, stanford_test, stanford_classes = files.process_stanford40()
    TVHI_train, TVHI_validation, TVHI_test, TVHI_classes = files.process_TVHI()

    # get the actual list of files for train, validation and test from stanford dataset from the file names
    #stanford_train_files, stanford_validation_files, stanford_test_files = load_data.load_stanford(
    #    stanford_train[0], stanford_validation[0], stanford_test[0])

    #load_data.load_tvhi(TVHI_train[0], TVHI_validation[0], TVHI_test[0])






if __name__ == "__main__":
    main()
