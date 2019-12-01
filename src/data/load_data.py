import pandas as pd


def load_wiki_df_from_csv(path='../../data/processed/wiki_df.csv'):
    columns = ['full_path', 'gender', 'age']
    wiki_df = pd.read_csv(path, sep=';')

    if wiki_df is None:
        raise Exception("Unable to open data frame")

    wiki_df = wiki_df[columns]

    for col in columns:
        if col not in wiki_df.columns:
            raise Exception("Column not found in dataframe columns")

    return wiki_df


def load_wiki_df_from_pkl(path='../../data/processed/wiki_meta_df.pkl'):
    columns = ['full_path', 'gender', 'age']
    wiki_df = pd.read_pickle(path)

    if wiki_df is None:
        raise Exception("Unable to open data frame")

    wiki_df = wiki_df[columns]

    for col in columns:
        if col not in wiki_df.columns:
            raise Exception("Column not found in dataframe columns")

    return wiki_df


def load_imdb_df_from_csv(path='../../data/processed/imdb_df.csv'):
    columns = ['full_path', 'gender', 'age']
    imdb_df = pd.read_csv(path, sep=';')

    if imdb_df is None:
        raise Exception("Unable to open data frame")

    imdb_df = imdb_df[columns]

    for col in columns:
        if col not in imdb_df.columns:
            raise Exception("Column not found in dataframe columns")

    return imdb_df


def load_imdb_df_from_pkl(path='../../data/processed/imdb_meta_df.pkl'):
    columns = ['full_path', 'gender', 'age']
    imdb_df = pd.read_pickle(path)

    if imdb_df is None:
        raise Exception("Unable to open data frame")

    imdb_df = imdb_df[columns]

    for col in columns:
        if col not in imdb_df.columns:
            raise Exception("Column not found in dataframe columns")

    return imdb_df
