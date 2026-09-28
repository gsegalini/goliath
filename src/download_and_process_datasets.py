import json
import os
import zipfile
from urllib import request
import numpy as np
import pandas as pd
from scipy.io import arff

DATA_DIR = "datasets"

NAME_URL_DICT_UCI = {
    "abalone": "https://archive.ics.uci.edu/static/public/1/abalone.zip",
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "default": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "magic": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
    "shoppers": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "gesture": "https://archive.ics.uci.edu/static/public/302/gesture+phase+segmentation.zip",
    "letter": "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip",
    "bean": "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
    "australia": "Manual Kaggle download @ (https://www.kaggle.com/datasets/luisblanche/quarterly-tourism-in-australia)",
    "credit": "https://www.openml.org/data/v1/download/31/credit-g.arff",
    "census": "https://archive.ics.uci.edu/static/public/117/census+income+kdd.zip",
}

# Census-Income (KDD) column order, taken from the ordered attribute list in census-income.names
# (the alphabetical block earlier in that file is by internal code, not data order).
CENSUS_COLS = [
    "age", "class of worker", "detailed industry recode", "detailed occupation recode",
    "education", "wage per hour", "enroll in edu inst last wk", "marital stat",
    "major industry code", "major occupation code", "race", "hispanic origin", "sex",
    "member of a labor union", "reason for unemployment", "full or part time employment stat",
    "capital gains", "capital losses", "dividends from stocks", "tax filer stat",
    "region of previous residence", "state of previous residence",
    "detailed household and family stat", "detailed household summary in household",
    "instance weight", "migration code-change in msa", "migration code-change in reg",
    "migration code-move within reg", "live in this house 1 year ago",
    "migration prev res in sunbelt", "num persons worked for employer",
    "family members under 18", "country of birth father", "country of birth mother",
    "country of birth self", "citizenship", "own business or self employed",
    "fill inc questionnaire for veteran's admin", "veterans benefits", "weeks worked in year",
    "year", "income",
]

CATCOLS = {
    "abalone": ["sex"],
    "adult": [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "salary",
    ],
    "magic": ["class"],
    "shoppers": ["Month", "VisitorType", "Weekend", "Revenue"],
    "letter": ["lettr"],
    "bean": ["Class"],
    "gesture": ["32"],
    "default": ["SEX", "EDUCATION", "MARRIAGE"],
    "australia": ['year', 'month', 'Region', 'State', 'Purpose'],
    "credit": [
        "checking_status", "credit_history", "purpose", "savings_status",
        "employment", "personal_status", "other_parties", "property_magnitude",
        "other_payment_plans", "housing", "job", "own_telephone",
        "foreign_worker", "class",
    ],
    # Everything except the 7 documented continuous attributes (age, wage per hour, capital
    # gains, capital losses, dividends from stocks, num persons worked for employer, weeks
    # worked in year). "instance weight" is dropped in process_census() because the dataset
    # documentation marks it "ignore".
    #
    # Five of these are integer-coded and would be misread as numeric by dtype inference:
    # the two *recode* columns (52 and 47 levels), own business, veterans benefits, and year.
    # census-income.names lists all five under nominal, so they are categorical here.
    "census": [
        "class of worker", "detailed industry recode", "detailed occupation recode",
        "education", "enroll in edu inst last wk", "marital stat", "major industry code",
        "major occupation code", "race", "hispanic origin", "sex", "member of a labor union",
        "reason for unemployment", "full or part time employment stat", "tax filer stat",
        "region of previous residence", "state of previous residence",
        "detailed household and family stat", "detailed household summary in household",
        "migration code-change in msa", "migration code-change in reg",
        "migration code-move within reg", "live in this house 1 year ago",
        "migration prev res in sunbelt", "family members under 18", "country of birth father",
        "country of birth mother", "country of birth self", "citizenship",
        "own business or self employed", "fill inc questionnaire for veteran's admin",
        "veterans benefits", "year", "income",
    ],
}


def fetchColumnTypes(dataname, df):
    setAll = set(list(df.columns))
    cat_cols = CATCOLS[dataname]
    setCat = set(cat_cols)
    setNum = setAll.difference(setCat)
    return setCat, setNum


def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name):
    print(f"Start processing dataset {name} from UCI.")
    save_dir = f"{DATA_DIR}/{name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        if name == "australia":
            unzip_file(f'{DATA_DIR}/archive.zip',
                       save_dir)  # download the archive.zip for australia and place in datasets/
        elif name == "credit":
            url = NAME_URL_DICT_UCI[name]
            request.urlretrieve(url, f"{save_dir}/credit-g.arff")
            print(f"Finish downloading dataset from {url}, data has been saved to {save_dir}.")
        else:
            url = NAME_URL_DICT_UCI[name]
            request.urlretrieve(url, f"{save_dir}/{name}.zip")
            print(f"Finish downloading dataset from {url}, data has been saved to {save_dir}.")
            unzip_file(f"{save_dir}/{name}.zip", save_dir)
        print(f"Finish unzipping {name}.")
    else:
        print("Already downloaded.")


def process_gesture():
    file_names = ["a1_va3", "a2_va3", "a3_va3", "b1_va3", "b1_va3", "c1_va3", "c3_va3"]
    datas = []
    for name in file_names:
        df = pd.read_csv(f"{DATA_DIR}/gesture/{name}.csv")
        data = df.to_numpy()
        datas.append(data)
    data = np.concatenate(datas, axis=0)
    data_df = pd.DataFrame(data)
    data_df.to_csv(f"{DATA_DIR}/gesture/data.csv", index=False)


def process_letter():
    dataname = "letter"
    path = f"{DATA_DIR}/{dataname}/{dataname}-recognition.data"
    save_path = f"{DATA_DIR}/{dataname}/data.csv"
    df = pd.read_csv(
        path,
        header=None,
        names=[
            "lettr",
            "xbox",
            "ybox",
            "width",
            "height",
            "onpix",
            "xbar",
            "ybar",
            "x2bar",
            "y2bar",
            "xybar",
            "x2ybar",
            "xy2bar",
            "xeg",
            "xegvy",
            "yeg",
            "yegvx",
        ],
    )
    cols = df.columns.tolist()
    cols = cols[1:] + cols[:1]
    df = df[cols]
    df.to_csv(save_path, index=False, header=True)


def process_abalone():
    path = f"{DATA_DIR}/abalone/abalone.data"
    save_path = f"{DATA_DIR}/abalone/data.csv"
    data_df = pd.read_csv(
        path,
        header=None,
        names=[
            "sex",
            "length",
            "diameter",
            "height",
            "whole-weight",
            "shucked-weight",
            "viscera-weight",
            "shell-weight",
            "rings",
        ],
    )
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)


def process_adult():
    path = f"{DATA_DIR}/adult/adult.data"
    save_path = f"{DATA_DIR}/adult/data.csv"
    data_df = pd.read_csv(
        path,
        header=None,
        names=[
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "salary",
        ],
    )
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)


def process_shoppers():
    path = f"{DATA_DIR}/shoppers/online_shoppers_intention.csv"
    save_path = f"{DATA_DIR}/shoppers/data.csv"
    data_df = pd.read_csv(path)
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)


def process_default():
    path = f"{DATA_DIR}/default/default of credit card clients.xls"
    save_path = f"{DATA_DIR}/default/data.csv"
    data_df = pd.read_excel(path, sheet_name="Data", header=1)
    data_df = data_df.drop("ID", axis=1)
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)


def process_magic():
    path = f"{DATA_DIR}/magic/magic04.data"
    save_path = f"{DATA_DIR}/magic/data.csv"
    data_df = pd.read_csv(
        path,
        header=None,
        names=["length", "weight", "size", "conc", "conc1", "asym", "m3long", "m3trans", "alpha", "distance", "class"],
    )
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)


def process_bean():
    path = f"{DATA_DIR}/bean/DryBeanDataset/Dry_Bean_Dataset.xlsx"
    save_path = f"{DATA_DIR}/bean/data.csv"
    data_df = pd.read_excel(path, sheet_name="Dry_Beans_Dataset")
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)


def process_australia():
    path = f'{DATA_DIR}/australia/tourism.csv'
    save_path = f'{DATA_DIR}/australia/data.csv'
    data_df = pd.read_csv(path)
    data_df["date"] = pd.to_datetime(data_df["Quarter"])
    data_df["year"] = data_df["date"].dt.year
    data_df["month"] = data_df["date"].dt.month
    """Ignore the day bcs it is always the first day of the month for this dataset"""
    data_df = data_df.drop(columns=['Unnamed: 0', 'date', 'Quarter'])
    data_df.to_csv(save_path, index=False)


def process_credit():
    path = f"{DATA_DIR}/credit/credit-g.arff"
    save_path = f"{DATA_DIR}/credit/data.csv"
    data, meta = arff.loadarff(path)
    data_df = pd.DataFrame(data)
    # scipy.io.arff returns nominal columns as bytes; decode to strings
    for col in data_df.select_dtypes(include=[object]).columns:
        data_df[col] = data_df[col].str.decode("utf-8")
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)


def process_census():
    """Census-Income (KDD): 34 categorical + 7 continuous columns.

    Added for the rebuttal to answer the "many-category tables" weakness: adult, our most
    categorical dataset, has 9 categorical columns and 3 with cardinality >= 10, whereas this
    has 34 and 10 (up to 51 levels). Same census/income domain, so it reads as adult scaled up.

    The UCI zip contains a tarball, not the data directly. `.data` and `.test` are the
    publisher's split of one population; we concatenate and let train_test_split() re-split,
    matching how the other datasets are handled.
    """
    import tarfile

    base = f"{DATA_DIR}/census"
    tarball = f"{base}/census.tar.gz"
    if not os.path.exists(f"{base}/census-income.data"):
        os.chmod(tarball, 0o644)  # ships read-only for the owner, which breaks re-extraction
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(base)

    frames = [
        pd.read_csv(f"{base}/census-income.{part}", header=None, names=CENSUS_COLS,
                    skipinitialspace=True, na_values=["?"])
        for part in ("data", "test")
    ]
    data_df = pd.concat(frames, ignore_index=True)
    data_df = data_df.drop(columns=["instance weight"])  # documentation says to ignore it
    data_df["income"] = data_df["income"].str.rstrip(".").str.strip()
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(f"{base}/data.csv", index=False)


def train_test_split(dataname, ratio=0.7):
    data_dir = f"{DATA_DIR}/{dataname}"
    path = f"{DATA_DIR}/{dataname}/data.csv"
    data_df = pd.read_csv(path)
    cat_cols, num_cols = fetchColumnTypes(dataname, data_df)
    metadata = {"cat_cols": list(cat_cols), "num_cols": list(num_cols)}
    with open(f"{DATA_DIR}/{dataname}/info.json", "w") as f:
        json.dump(metadata, f, indent=4)
    total_num = data_df.shape[0]

    num_train = int(total_num * ratio)
    num_test = total_num - num_train
    seed = 42

    np.random.seed(seed)
    # don't shuffle for australia bcs it is a time series
    shuffled_ids = np.random.permutation(np.arange(total_num)) if dataname != 'australia' else np.arange(total_num)

    train_idx = shuffled_ids[:num_train]
    test_idx = shuffled_ids[-num_test:]

    train_df = data_df.loc[train_idx]
    test_df = data_df.loc[test_idx]

    train_path = f"{data_dir}/train.csv"
    test_path = f"{data_dir}/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Spliting Trainig and Testing data for {dataname} is done.")
    print(f"Training data shape: {train_df.shape}, Testing data shape: {test_df.shape}")
    print(f"Training data saved at {train_path}, Testing data saved at {test_path}.")


if __name__ == "__main__":
    # Downloading dataset
    for name in NAME_URL_DICT_UCI.keys():
        download_from_uci(name)
        globals()[f"process_{name}"]()

    for name in NAME_URL_DICT_UCI.keys():
        eval(f"process_{name}()")
        train_test_split(name, ratio=0.7)
