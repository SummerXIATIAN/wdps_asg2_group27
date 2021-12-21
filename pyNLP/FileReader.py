import pandas as pd


def fileReader(dataPath, targetFile,versionDescriptionFile):
    df = pd.read_csv(dataPath+targetFile)
    print('There are {} rows and {} columns in dataset: {}'.format(
        df.shape[0], df.shape[1], targetFile))
    with open(dataPath+versionDescriptionFile, 'r') as file:
        data = file.read()
    return df,data
