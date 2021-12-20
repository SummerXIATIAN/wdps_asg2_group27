import pandas as pd


def fileReader(dataPath, targetFile):
    df = pd.read_csv(dataPath+targetFile)
    print('There are {} rows and {} columns in dataset: {}'.format(
        df.shape[0], df.shape[1], targetFile))
    return df
