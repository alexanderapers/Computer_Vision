
def main():
    with open('Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
        print(f'Train files ({len(train_files)}):\n\t{train_files}')
        print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')

    with open('Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
        print(f'Test files ({len(test_files)}):\n\t{test_files}')
        print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')
    
    action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))


    print(f'Action categories ({len(action_categories)}):\n{action_categories}')
    print("Hello world!")

if __name__ == "__main__":
    main()