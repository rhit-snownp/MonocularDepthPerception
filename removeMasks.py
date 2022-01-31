import os
for subdir, dirs, files in os.walk("images/train"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith("mask.npy"):
            os.remove(filepath)
            print(filepath)