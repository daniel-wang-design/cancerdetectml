import os
import random
import shutil

seed = 1
random.seed(seed)
images_directory = "data/images/"
training_directory = "data/train/"
test_directory = "data/test/"
validation_directory = "data/validation/"

print("Creating required folders")

os.makedirs(training_directory + "benign/")
os.makedirs(training_directory + "malignant/")
os.makedirs(test_directory + "benign/")
os.makedirs(test_directory + "malignant/")
os.makedirs(validation_directory + "benign/")
os.makedirs(validation_directory + "malignant/")

test_examples = train_examples = validation_examples = 0

print("Begin copying")

for line in open("data/training.csv").readlines()[1:]:
    split_line = line.split(",")
    img_file = split_line[0]
    benign_malign = split_line[1]

    random_num = random.random()
    if random_num < 0.5:
        continue
    if random_num < 0.8:
        location = training_directory
        train_examples += 1

    elif random_num < 0.9:
        location = validation_directory
        validation_examples += 1

    else:
        location = test_directory
        test_examples += 1

    if int(float(benign_malign)) == 0:
        shutil.copy(images_directory + img_file + ".jpg", location + "benign/" + img_file + ".jpg", )

    elif int(float(benign_malign)) == 1:
        shutil.copy(images_directory + img_file + ".jpg", location + "malignant/" + img_file + ".jpg", )
    print(f"File {img_file} copied")

print("Copy complete")
print(f"Number of training examples {train_examples}")
print(f"Number of test examples {test_examples}")
print(f"Number of validation examples {validation_examples}")
