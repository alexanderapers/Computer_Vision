from sklearn.model_selection import train_test_split


def get_stanford40_splits():
    with open('Stanford40/ImageSplits/actions.txt') as f:
        action_categories = f.readlines()[1:]
        action_categories = [action.split()[0] for action in action_categories]
        action_dict = { action: idx for idx, action in enumerate(action_categories) }

    with open('Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = [action_dict['_'.join(name.split('_')[:-1])] for name in train_files]

        train_files, validation_files, train_labels, validation_labels = train_test_split(
            train_files, train_labels, train_size=0.9, random_state=42,
            shuffle=True, stratify=train_labels)

        train = (train_files, train_labels)
        validation = (validation_files, validation_labels)

    with open('Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = [action_dict['_'.join(name.split('_')[:-1])] for name in test_files]

        test = (test_files, test_labels)

    return train, validation, test, action_categories


def get_tvhi_splits():
    set_1_indices = [[2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
                 [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
                 [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
                 [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42]]
    set_2_indices = [[1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
                 [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
                 [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
                 [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50]]

    classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

    # test set
    set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
    # print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
    # print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')

    test = (set_1, set_1_label)

    # training set
    set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
    # print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
    # print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')

    # Use 10% of training data for validation, use stratification
    train_files, validation_files, train_labels, validation_labels = train_test_split(
        set_2, set_2_label, train_size=0.9, random_state=42, shuffle=True,
        stratify=set_2_label)

    train = (train_files, train_labels)
    validation = (validation_files, validation_labels)

    return train, validation, test, classes
