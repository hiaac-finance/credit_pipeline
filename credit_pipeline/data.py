import os
import pandas as pd
import credit_pipeline.data_exploration as dex


def download_datasets():
    import gdown
    """Download data from Google Drive and unzip it in the data folder."""
    url = "https://drive.google.com/uc?id=1Y7bTNsxDv-te40FnJsoca1YeB4da6TCq"
    output = "data.zip"
    gdown.download(url, output, quiet=False)
    os.system("unzip data.zip -d data")
    os.system("rm data.zip")


def prepare_datasets():
    """
    Function that preprocess the datasets and save them in the data/prepared folder.
    """
    # make dir if not exist
    os.makedirs("data/prepared", exist_ok=True)

    # home credit
    df = dex.read_csv_encoded("data/HomeCredit", "application_train.csv")
    df = df.drop(columns=["SK_ID_CURR", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"])
    df = df.rename(columns={"TARGET": "DEFAULT"})
    cat_cols = df.loc[:, df.dtypes == "object"].columns.tolist()
    for col in cat_cols:
        df[col] = pd.Categorical(df[col])
    df.to_csv("data/prepared/homecredit.csv", index=False)

    # taiwan
    df = dex.read_csv_encoded("data/Taiwan/", "Taiwan.csv")
    df.columns = df.iloc[0, :].tolist()
    df = df.iloc[1:, :]
    df = df.drop(columns=["ID"])
    df = df.rename(columns={"default payment next month": "DEFAULT"})
    df = df.astype("float64")
    sex_map = {2: "Female", 1: "Male"}
    education_map = {
        -2: "Unknown",
        -1: "Unknown",
        0: "Unknown",
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Others",
        5: "Unknown",
        6: "Unknown",
    }
    marriage_map = {
        0: "Others",
        1: "Married",
        2: "Single",
        3: "Others",
    }
    df.SEX = df.SEX.apply(sex_map.get)
    df.EDUCATION = df.EDUCATION.apply(education_map.get)
    df.MARRIAGE = df.MARRIAGE.apply(marriage_map.get)
    cat_cols = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col])
    df.to_csv("data/prepared/taiwan.csv", index=False)

    # german
    df = dex.read_csv_encoded("data/German/", "german.csv")
    df.columns = [
        "CheckingAccount",
        "Duration",
        "CreditHistory",
        "Purpose",
        "CreditAmount",
        "SavingsAccount",
        "EmploymentSince",
        "InstallmentRate",
        "PersonalStatus",
        "OtherDebtors",
        "ResidenceSince",
        "Property",
        "Age",
        "OtherInstallmentPlans",
        "Housing",
        "ExistingCredits",
        "Job",
        "Dependents",
        "Telephone",
        "ForeignWorker",
        "DEFAULT",
    ]
    df["DEFAULT"] = df.DEFAULT.apply(lambda x: 1 if x == 2 else 0)
    df["Gender"] = df.PersonalStatus
    df.CheckingAccount = df.CheckingAccount.apply(
        {"A11": "< 0", "A12": "0 - 200", "A13": "> 200", "A14": "No"}.get
    )
    df.CreditHistory = df.CreditHistory.apply(
        {
            "A30": "No credits/all paid",
            "A31": "All paid",
            "A32": "Existing paid",
            "A33": "Delay in paying",
            "A34": "Critical account",
        }.get
    )
    df.Purpose = df.Purpose.apply(
        {
            "A40": "Car (new)",
            "A41": "Car (used)",
            "A42": "Furniture/equipment",
            "A43": "Radio/television",
            "A44": "Domestic appliances",
            "A45": "Repairs",
            "A46": "Education",
            "A47": "Vacation",
            "A48": "Retraining",
            "A49": "Business",
            "A410": "Others",
        }.get
    )
    df.SavingsAccount = df.SavingsAccount.apply(
        {
            "A61": "< 100",
            "A62": "100 - 500",
            "A63": "500 - 1000",
            "A64": "> 1000",
            "A65": "Unknown/None",
        }.get
    )
    df.EmploymentSince = df.EmploymentSince.apply(
        {
            "A71": "Unemployed",
            "A72": "< 1",
            "A73": "1 - 4",
            "A74": "4 - 7",
            "A75": "> 7",
        }.get
    )
    df.Gender = df.Gender.apply(
        {
            "A91": "Male",
            "A92": "Female",
            "A93": "Male",
            "A94": "Male",
            "A95": "Female",
        }.get
    )
    df.OtherDebtors = df.OtherDebtors.apply(
        {"A101": "No", "A102": "Co-applicant", "A103": "Guarantor"}.get
    )
    df.Property = df.Property.apply(
        {
            "A121": "Real estate",
            "A122": "Savings agreement/life insurance",
            "A123": "Car or other",
            "A124": "Unknown/None",
        }.get
    )
    df.OtherInstallmentPlans = df.OtherInstallmentPlans.apply(
        {"A141": "Bank", "A142": "Stores", "A143": "No"}.get
    )
    df.Housing = df.Housing.apply(
        {"A151": "Rent", "A152": "Own", "A153": "For free"}.get
    )
    df.Job = df.Job.apply(
        {
            "A171": "Unemployed",
            "A172": "Unskilled",
            "A173": "Skilled",
            "A174": "Highly skilled",
        }.get
    )
    df.Telephone = df.Telephone.apply({"A191": 0, "A192": 1}.get)
    df.ForeignWorker = df.ForeignWorker.apply({"A201": 1, "A202": 0}.get)
    df = df.drop(columns=["PersonalStatus"])
    cat_cols = [
        "CheckingAccount",
        "CreditHistory",
        "Purpose",
        "SavingsAccount",
        "EmploymentSince",
        "Gender",
        "OtherDebtors",
        "Property",
        "OtherInstallmentPlans",
        "Housing",
        "Job",
        "Telephone",
        "ForeignWorker",
    ]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col])
    df.to_csv("data/prepared/german.csv", index=False)


def load_dataset(dataset_name):
    """Function to load the prepared datasets, includes
    Home Credit, Taiwan and German.

    :param dataset_name: string, name of the dataset
    :return: pandas dataframe
    """
    if dataset_name == "homecredit":
        df = pd.read_csv("../data/prepared/homecredit.csv")
        cat_cols = df.loc[:, df.dtypes == "object"].columns.tolist()
    elif dataset_name == "taiwan":
        df = pd.read_csv("../data/prepared/taiwan.csv")
        cat_cols = [
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ]
    elif dataset_name == "german":
        df = pd.read_csv("../data/prepared/german.csv")
        cat_cols = [
            "CheckingAccount",
            "CreditHistory",
            "Purpose",
            "SavingsAccount",
            "EmploymentSince",
            "Gender",
            "OtherDebtors",
            "Property",
            "OtherInstallmentPlans",
            "Housing",
            "Job",
            "Telephone",
            "ForeignWorker",
        ]

    for col in cat_cols:
        df[col] = pd.Categorical(df[col])
    return df


if __name__ == "__main__":
    download_datasets()
    prepare_datasets()
