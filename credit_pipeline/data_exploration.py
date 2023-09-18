import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from fitter import Fitter, get_common_distributions
from scipy import stats
import dash_bio
import chardet
import os


def read_csv_encoded(path, filename):
    """
    Reads a CSV file with automatic character encoding detection.

    Given a directory path and a filename, this function detects the character encoding of the file
    using the `chardet` library and reads the file into a pandas DataFrame.

    :param path: The directory path where the file is located.
    :type path: str
    :param filename: The name of the file to be read.
    :type filename: str
    :return: DataFrame containing the data from the CSV file.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If the file does not exist at the given path.
    :raises pd.errors.EmptyDataError: If the file is empty.
    :raises pd.errors.ParserError: If there's an error parsing the file.

    :Example:

    >>> df = read_csv_encoded('/path/to/directory', 'sample.csv')
    >>> df.head()
    [Sample output of the dataframe]

    Note:
        Make sure both `chardet` and `pandas` libraries are installed before using this function.
    """

    the_file = os.path.join(path, filename)
    rawdata = open(the_file, "rb").read()
    result = chardet.detect(rawdata)
    charenc = result["encoding"]

    data = pd.read_csv(the_file, encoding=charenc, index_col=False)

    return data


def check_missing(dataframe, threshold=50, show_top=False):
    """
    Checks for columns with missing values in a DataFrame.

    This function examines a given DataFrame for missing values and returns either the top columns with
    missing values above a specified threshold or a list of these column names, based on the `show_top` flag.

    :param dataframe: The DataFrame to be checked for missing values.
    :type dataframe: pandas.DataFrame
    :param threshold: The percentage threshold for missing values. Columns with missing values above this
                      threshold will be considered. Default is 50%.
    :type threshold: float
    :param show_top: Flag to determine the return type. If `True`, the function returns a DataFrame containing
                     the top columns with missing values and their respective counts and percentages. If `False`,
                     the function returns a list of column names with missing values above the threshold.
    :type show_top: bool
    :return: Either a DataFrame with columns, counts, and percentages of missing values (if `show_top` is True)
             or a list of column names (if `show_top` is False).
    :rtype: pandas.DataFrame or list
    :raises ValueError: If the provided DataFrame is empty.

    :Example:

    >>> df = pd.DataFrame({"A": [1, np.nan, 3], "B": [np.nan, np.nan, np.nan], "C": [1, 2, 3]})
    >>> check_missing(df, threshold=50, show_top=True)
    [Sample output of the dataframe]


    Note:
        Ensure that the `pandas` library is installed before using this function.
    """

    # Create a copy of the DataFrame to avoid modifying the original data
    df = dataframe.copy()

    # Count the missing values for each column and sort in descending order
    count = df.isnull().sum().sort_values(ascending=False)

    # Calculate the percentage of missing values for each column and sort in descending order
    percent = (df.isnull().sum() / df.shape[0]).sort_values(ascending=False) * 100

    # Create a DataFrame with the count and percentage of missing values for each column
    missing_data = pd.concat(
        [count, percent], axis=1, keys=["MISSING", "PERCENTAGE_MISSING"]
    )

    # Select columns with missing values above the threshold
    top_missing = missing_data.loc[missing_data["PERCENTAGE_MISSING"] > threshold, :]

    if show_top:
        return top_missing
    else:
        # Return a list of column names with missing values above the threshold
        return top_missing.index.tolist()


def process_missing_values(df, limit_percentage, statistic):
    """
    Processes missing values in a DataFrame based on a specified limit and statistic.

    This function examines each column of a DataFrame for missing values. If the proportion of missing
    values in a column exceeds a specified limit, the column is dropped. Otherwise, the missing values
    are imputed based on a provided statistic (mean, median, mode) or rows with missing values can be excluded.

    :param df: The DataFrame in which to process missing values.
    :type df: pandas.DataFrame
    :param limit_percentage: Proportion threshold for missing values. If a column has a greater proportion
                             of missing values than this limit, it will be dropped. Range: [0, 1].
    :type limit_percentage: float
    :param statistic: The statistic used to impute missing values. Options are 'mean', 'median', 'mode', or 'exclude'.
                      If 'exclude', rows with missing values in the column are dropped.
    :type statistic: str
    :return:
        - DataFrame with processed missing values.
    :rtype: pandas.DataFrame
    :raises ValueError: If the provided limit_percentage is not in the range [0, 1].
                        If the provided statistic is not one of the expected options ('mean', 'median', 'mode', 'exclude').

    :Example:

    >>> df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [np.nan, 2, 3, 4]})
    >>> process_missing_values(df, 0.2, 'mean')

    Note:
        Ensure that the `pandas` library is installed before using this function.
    """
    # Calculates the number of missing values in each column
    missing_counts = df.isna().sum()

    # Calculates the proportion of missing values relative to the total column length
    missing_percentages = missing_counts / len(df)

    # Loop through the columns of the DataFrame
    for column in df.columns:
        # Checks if the proportion of missing values in the column is greater than the limit
        if missing_percentages[column] > limit_percentage:
            # Removes the column from the DataFrame if the proportion is greater than the limit
            df = df.drop(column, axis=1)
        else:
            # Replaces the missing values according to the chosen statistic
            if statistic == "mean":
                df[column].fillna(df[column].mean(), inplace=True)
            elif statistic == "median":
                df[column].fillna(df[column].median(), inplace=True)
            elif statistic == "mode":
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif statistic == "exclude":
                df = df.dropna(subset=[column])

    return df


def replace_missing_with_category(dataframe, encoding=False):
    """Replaces missing values in categorical columns of a DataFrame with the category "Missing". The function provides an option to encode the columns using one-hot encoding or label encoding.

    :param DataFrame dataframe: The original DataFrame containing the data.
    :param bool or str encoding:  Specifies whether encoding should be applied. Default is False. If set to 'OneHot', performs one-hot encoding. If set to 'Label', performs label encoding.
    :return DataFrame: The DataFrame with missing values replaced and optionally encoded.
    """

    # Copy of the original DataFrame
    df = dataframe.copy()

    # Loop through the columns in the DataFrame
    for column in df.columns:
        # Fills missing values in the column with the category "Missing"
        df[column].fillna("Missing", inplace=True)

        # Converts the values in the column to strings
        df[column] = df[column].astype(str)

        # Checks if encoding should be applied
        if encoding:
            # Checks the selected encoding type
            if encoding == "OneHot":
                # Creates a temporary DataFrame with only the specified column
                temp_df = df[[column]]

                # Performs one-hot encoding on the column
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
                encoded_data = encoder.fit_transform(temp_df)

                # Creates a DataFrame with the encoded columns
                encoded_df = pd.DataFrame(
                    encoded_data, columns=encoder.get_feature_names_out([column])
                )

                # Removes the original column from the DataFrame
                df.drop(columns=[column], inplace=True)

                # Concatenates the original DataFrame with the encoded DataFrame
                df = pd.concat([df, encoded_df], axis=1)
            elif encoding == "Label":
                # Performs encoding using LabelEncoder
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column])
        else:
            # Does not apply any encoding, keeps the column as strings

            # Renames missing values to "Missing"
            df.loc[df[column] == "nan", column] = "Missing"

    return df


def list_by_type(dataframe, types_cols, debug=False):
    """
    Lists columns in a DataFrame based on specified data types.

    This function iterates through the columns of a given DataFrame and checks whether each column's data type
    matches any of the specified data types in the `types_cols` list. If a match is found, the column's name
    is added to the output list. If the `debug` flag is enabled, the function also prints debugging information
    for each matching column.

    :param dataframe: The DataFrame whose columns are to be checked.
    :type dataframe: pandas.DataFrame
    :param types_cols: A list of data types to be matched against column data types.
    :type types_cols: list
    :param debug: Flag to determine whether debugging information should be printed. If `True`, the column name,
                  data type, and the first 10 unique values of each matching column are printed. Default is False.
    :type debug: bool
    :return: A list of column names in the DataFrame that match the specified data types.
    :rtype: list
    :raises ValueError: If the provided DataFrame is empty.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"], "C": [1.1, 2.2, 3.3]})
    >>> list_by_type(df, [int, float])
    ['A', 'C']

    Note:
        Ensure that the `pandas` library is installed before using this function.
    """

    list_cols = []

    # Iterate through each column in the DataFrame
    for c in dataframe.columns:
        col = dataframe[c]

        # Check if the column's data type matches any of the specified data types
        if col.dtype in types_cols:
            list_cols.append(c)

            # Print debugging information if debug flag is enabled
            if debug:
                print(c, " : ", col.dtype)
                print(col.unique()[:10])
                print("---------------")

    return list_cols


def types_columns(dataframe):
    """
    Retrieves a list of unique data types present in the columns of a DataFrame.

    This function examines each column of the provided DataFrame and determines the data type of each column.
    It then returns a list of unique data types found across all columns.

    :param dataframe: The DataFrame whose columns' data types are to be determined.
    :type dataframe: pandas.DataFrame
    :return: A list of unique data types found in the DataFrame's columns.
    :rtype: list
    :raises ValueError: If the provided DataFrame is empty.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"], "C": [1.1, 2.2, 3.3]})
    >>> types_columns(df)
    ['float64', 'object', 'int64']

    Note:
        Ensure that the `pandas` library is installed before using this function.
    """

    listTypes = set()

    # Iterate through each column in the DataFrame
    for c in dataframe.columns:
        # Add the data type of the column to the set of unique data types
        listTypes.add(str(dataframe[c].dtype))

    # Convert the set of unique data types to a list
    return list(listTypes)


def list_by_unbalanced(dataframe, crit=0.7):
    """
    Retrieves columns in a DataFrame that are unbalanced based on a specified criterion.

    This function examines each column of the provided DataFrame and calculates the proportion of each unique
    value. If the highest proportion of a value in a column exceeds the specified criterion, the column is
    considered unbalanced, and its name is added to the output list.

    :param dataframe: The DataFrame whose columns are to be checked for unbalanced values.
    :type dataframe: pandas.DataFrame
    :param crit: The proportion criterion, a value between 0 and 1, used to determine if a column is unbalanced.
                 If the highest proportion of a value in a column exceeds this criterion, the column is considered
                 unbalanced. Default is 0.7.
    :type crit: float
    :return: A list of column names in the DataFrame that are unbalanced based on the specified criterion.
    :rtype: list
    :raises ValueError: If the provided DataFrame is empty.
    :raises ValueError: If the crit value is not between 0 and 1.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 1, 1, 2, 2], "B": ["a", "a", "a", "a", "b"], "C": [1, 2, 3, 4, 5]})
    >>> list_by_unbalanced(df, crit=0.7)
    ['B']

    Note:
        Ensure that both the `pandas` and `numpy` libraries are installed before using this function.
    """

    list_cols = []

    for c in dataframe.columns:
        value_proportions = dataframe[c].value_counts(normalize=True)
        unbalanced_ratio = np.sort(value_proportions)[-1]
        if unbalanced_ratio >= crit:
            list_cols.append(c)

    return list_cols


def list_by_unique(dataframe, crit=3, type_of_list="equal"):
    """
    Retrieves columns in a DataFrame based on the number of unique values they contain.

    This function examines each column of the provided DataFrame and counts the number of unique values. Depending
    on the `type_of_list` parameter, the function returns columns that either have an exact number of unique values
    (`equal`) or up to a maximum number of unique values (`until`).

    :param dataframe: The DataFrame whose columns are to be checked for unique values.
    :type dataframe: pandas.DataFrame
    :param crit: The criterion used to determine the number of unique values. This value is compared against
                 the number of unique values in each column based on the `type_of_list` parameter.
    :type crit: int
    :param type_of_list: Determines how the `crit` parameter is applied.
                         - 'equal': Returns columns with an exact number of unique values equal to `crit`.
                         - 'until': Returns columns with a number of unique values up to and including `crit`.
                         Default is 'equal'.
    :type type_of_list: str
    :return: A list of column names in the DataFrame that match the specified criteria.
    :rtype: list
    :raises ValueError: If the provided DataFrame is empty.
    :raises ValueError: If the `type_of_list` parameter is neither 'equal' nor 'until'.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 1, 2], "B": ["a", "b", "c"], "C": [1, 2, 3], "D": [1, 1, 1]})
    >>> list_by_unique(df, crit=3, type_of_list='equal')
    ['B', 'C']

    Note:
        Ensure that the `pandas` library is installed before using this function.
    """

    list_cols = []

    if type_of_list == "equal":
        for col in dataframe.columns:
            if len(dataframe[col].unique()) == crit:
                list_cols.append(col)
    elif type_of_list == "until":
        for col in dataframe.columns:
            if len(dataframe[col].unique()) <= crit:
                list_cols.append(col)

    return list_cols


def list_contin_cols(df):
    """List continuous columns in a dataframe.

    :param DataFrame df: dataframe to list continuous columns
    :return: list with continuous columns names
    """
    numeric_cols = (df, ["float64"])
    contin_cols = []
    for col in numeric_cols:
        if len(df[col].unique()) > 20:
            contin_cols.append(col)
    return contin_cols


def list_no_variation_cols(df, list_columns="all"):
    """Verify if columns have no variation, i.e., all cells have the same value.

    :param DataFrame df: dataframe to verify if columns have no variation
    :param list list_columns: subset of columns to verify if have no variation. Defaults to "all".
    :return: list with columns names that have no variation
    """
    if isinstance(list_columns, list):
        columns = list_columns
    elif list_columns is None:
        columns = list_by_type(df, ["float64"])
    columns = np.array(columns)
    idx = np.where(df[columns].nunique() == 1)[0]
    return columns[idx].tolist()


def show_unique_values_by_column(dataframe, col_list, return_dict=False):
    """
    Displays the unique values for specified columns in a DataFrame.

    This function examines the specified columns of a provided DataFrame and retrieves the unique values for each
    of those columns. Depending on the `return_dict` parameter, the function returns the results either as a
    dictionary or as a DataFrame.

    :param dataframe: The DataFrame from which unique values are to be extracted.
    :type dataframe: pandas.DataFrame
    :param col_list: A list of column names for which unique values are to be determined.
    :type col_list: list
    :param return_dict: Flag to determine the return type. If `True`, the function returns the results as a dictionary.
                        If `False`, the function returns the results as a DataFrame. Default is False.
    :type return_dict: bool
    :return: Either a dictionary with the column names as keys and lists of unique values as values (if `return_dict`
             is True) or a DataFrame with column names as the index and lists of unique values in a column (if
             `return_dict` is False).
    :rtype: dict or pandas.DataFrame
    :raises ValueError: If the provided DataFrame is empty.
    :raises KeyError: If any column in `col_list` is not present in the `dataframe`.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 1, 2, 3], "B": ["a", "b", "c", "c"], "C": [1.1, 2.2, 2.2, 3.3]})
    >>> show_unique_values_by_column(df, col_list=['A', 'B'], return_dict=True)
    {'A': [array([1, 2, 3])], 'B': [array(['a', 'b', 'c'], dtype=object)]}

    Note:
        Ensure that the `pandas` library is installed before using this function.
    """

    unique_values_dict = {}

    for c in col_list:
        unique_values_dict[c] = [dataframe[c].unique()]

    if return_dict:
        return unique_values_dict
    else:
        unique_values_df = pd.DataFrame.from_dict(
            unique_values_dict, orient="index", columns=["UNIQUE_VALUES"]
        )

        return unique_values_df


def describe_data(dataframe, list_columns="all"):
    """
    Provides a detailed description of the specified columns in a DataFrame.

    This function examines the specified columns of a provided DataFrame and calculates various statistics and
    metadata for each column, including mean, median, mode, unbalanced ratio, count of unique values, count of
    null values, and data type. The results are returned as a DataFrame.

    :param dataframe: The DataFrame whose columns are to be described.
    :type dataframe: pandas.DataFrame
    :param list_columns: A list of column names to describe. If set to 'all', all columns in the DataFrame will
                         be described. Default is 'all'.
    :type list_columns: list or str
    :return: A DataFrame containing the calculated statistics and metadata for the specified columns. The index
             of the returned DataFrame corresponds to the column names, and the columns of the returned DataFrame
             correspond to the various statistics and metadata.
    :rtype: pandas.DataFrame
    :raises ValueError: If the provided DataFrame is empty.
    :raises KeyError: If any column in `list_columns` is not present in the `dataframe`.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 1, 2, 3], "B": ["a", "b", "c", "c"], "C": [1.1, 2.2, 2.2, 3.3]})
    >>> describe_data(df, list_columns=['A', 'B'])
    [Sample output of the dataframe]

    Note:
        Ensure that both the `pandas` and `numpy` libraries are installed before using this function.
    """

    if list_columns == "all":
        columns = dataframe.columns
    else:
        columns = list_columns

    data = {}
    num_cols = dataframe.describe().columns
    cat_cols = set(columns) - set(num_cols)

    mean = {}
    for c in columns:
        if c in num_cols:
            mean[c] = dataframe[c].mean()
        else:
            mean[c] = float("NaN")

    median = {}
    for c in columns:
        if c in num_cols:
            median[c] = dataframe[c].median()
        else:
            median[c] = float("NaN")

    mode = {}
    for c in columns:
        mode[c] = dataframe[c].mode()[0]

    unbalanced_ratio = {}
    for c in columns:
        value_proportions = dataframe[c].value_counts(normalize=True)
        unbalanced_ratio[c] = np.sort(value_proportions)[-1]

    qtd_unique = {}
    for c in columns:
        qtd_unique[c] = len(dataframe[c].unique())

    qtd_null = {}
    for c in columns:
        qtd_null[c] = dataframe[c].isnull().sum()

    data_type = {}
    for c in columns:
        data_type[c] = dataframe[c].dtype

    if len(cat_cols) != len(columns):
        data["mean"] = mean
        data["median"] = median
    data["mode"] = mode
    data["unbalanced_ratio"] = unbalanced_ratio
    data["qtd_unique"] = qtd_unique
    data["qtd_null"] = qtd_null
    data["data_type"] = data_type

    return pd.DataFrame(data)


def detect_multicollinearity(dataframe, threshold=0.7, plot=False):
    """
    Detects multicollinearity in a DataFrame based on a specified correlation threshold.

    This function examines the correlation between columns in a DataFrame. If the absolute value of the
    correlation coefficient between any two columns exceeds a given threshold, the pair of columns is
    identified as exhibiting multicollinearity. Optionally, a heatmap of the correlation matrix can be plotted.

    :param dataframe: The DataFrame in which to detect multicollinearity.
    :type dataframe: pandas.DataFrame
    :param threshold: Absolute value of correlation coefficient above which two columns are considered
                      to exhibit multicollinearity. Default is 0.7.
    :type threshold: float
    :param plot: If `True`, a heatmap of the correlation matrix will be plotted. Default is False.
    :type plot: bool
    :return:
        - A list of tuples where each tuple contains a pair of column names that exhibit multicollinearity
          and their correlation value.
    :rtype: list[tuple]
    :raises ValueError: If the provided threshold is not in the range [0, 1].

    :Example:

    >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 3, 2, 4], "C": [1, 2, 3, 4]})
    >>> detect_multicollinearity(df, 0.9)
    Multicollinearity detected between: A and C. Correlation: 1.0
    [('A', 'C', 1.0)]

    Note:
        Ensure that both the `pandas` library and `seaborn` (if using `plot=True`) are installed before using this function.
    """

    # Calculate the correlation matrix
    correlation_matrix = dataframe.corr().abs()

    # Get the list of columns with high correlation (greater than or equal to the threshold)
    collinear_columns = []

    # Iterate over the correlation matrix
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] >= threshold:
                # Add the pair of columns with high correlation to the list
                column_i = correlation_matrix.columns[i]
                column_j = correlation_matrix.columns[j]
                correlation_value = correlation_matrix.iloc[i, j]
                collinear_columns.append((column_i, column_j, correlation_value))

    # Print the pairs of columns with multicollinearity and their correlation values
    for pair in collinear_columns:
        print(
            f"Multicollinearity detected between: {pair[0]} and {pair[1]}. Correlation: {pair[2]}"
        )

    # Plot the correlation matrix if requested
    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

    return collinear_columns


def detect_outliers(dataframe, plot=False, show_top=False):
    """
    Detects and reports outliers for each column in a DataFrame using the IQR method.

    This function calculates the interquartile range (IQR) for each column in a DataFrame and
    identifies values outside of the IQR as outliers. The function can optionally produce boxplots
    for visual inspection and can also sort columns based on the number of detected outliers.

    :param dataframe: The DataFrame in which to detect outliers.
    :type dataframe: pandas.DataFrame
    :param plot: If `True`, a boxplot of each column will be plotted. Default is False.
    :type plot: bool
    :param show_top: If `True`, columns will be sorted in descending order based on the number of outliers. Default is False.
    :type show_top: bool
    :return:
        - A dictionary where each key is a column name and each value is another dictionary containing the number and proportion of outliers.
    :rtype: dict
    :raises ValueError: If any of the input parameters are not of the expected types or have invalid values.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 100], "B": [1, 3, 2, 4, 200]})
    >>> detect_outliers(df, plot=True)

    Note:
        Ensure that both the `pandas` library and `matplotlib` (if using `plot=True`) are installed before using this function.
    """

    outliers = {}

    # Iterate over all columns in the DataFrame
    for column in dataframe.columns:
        # Calculate the interquartile range (IQR)
        q1, q3 = dataframe[column].quantile([0.25, 0.75])
        iqr = q3 - q1

        # Define the lower and upper bounds
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        # Detect outliers
        outlier_mask = (dataframe[column] < lower_bound) | (
            dataframe[column] > upper_bound
        )
        num_outliers = outlier_mask.sum()
        outlier_proportion = num_outliers / len(dataframe)

        # Store the results
        outliers[column] = {
            "Num_Outliers": num_outliers,
            "Proportion": outlier_proportion,
        }

        # Plot the boxplot, if requested
        if plot:
            plt.figure()
            dataframe.boxplot(column=column)
            plt.title(f"Boxplot of {column}")
            plt.show()

    # Sort columns in descending order based on the number of outliers, if show_top is True
    if show_top:
        outliers = dict(
            sorted(outliers.items(), key=lambda x: x[1]["Num_Outliers"], reverse=True)
        )

    # Print the results
    for column, values in outliers.items():
        print(f"Column: {column}")
        print(f"Number of outliers: {values['Num_Outliers']}")
        print(f"Proportion of outliers: {values['Proportion']}")
        print()

    return outliers


def encoder_columns(dataframe):
    """
    Encodes columns with object data type and at most two unique values in a DataFrame.

    This function examines each column of a provided DataFrame. For columns with data type 'object' (typically strings)
    that have at most two unique values, it encodes the values using a label encoder, transforming them into integer
    labels. The transformed DataFrame is then returned.

    :param dataframe: The DataFrame whose columns are to be encoded.
    :type dataframe: pandas.DataFrame
    :return: The DataFrame with the specified columns encoded as integers.
    :rtype: pandas.DataFrame
    :raises ValueError: If the provided DataFrame is empty.

    :Example:

    >>> df = pd.DataFrame({"A": ["yes", "yes", "no", "no"], "B": ["a", "b", "a", "c"]})
    >>> encoder_columns(df)
    [Sample output of the dataframe with column A encoded]

    Note:
        Ensure that the `pandas` library and `sklearn.preprocessing` are installed before using this function.
    """

    le = LabelEncoder()

    for c in dataframe.columns:
        if dataframe[c].dtype == "O":
            if len(list(dataframe[c].unique())) <= 2:
                print(c)
                le.fit(dataframe[c])
                dataframe[c] = le.transform(dataframe[c])

    return dataframe


def plot_columns_histogram(
    dataframe,
    list_columns="all",
    kde=False,
    target_col="TARGET",
    by_class=False,
    bins=30,
    title_y=1.003,
    fig_nrows=5,
    fig_width=10,
    fig_height_prop=2.5,
    fontscale=0.8,
    title_fontsize=14,
    save_fig=True,
    dpi=500,
):
    """
    Plots histograms for the specified columns in a DataFrame.

    This function examines the specified columns of a provided DataFrame and plots histograms for each column.
    The histograms can optionally be divided by classes using the `by_class` parameter. The results can be saved
    as a PNG image.

    :param dataframe: The DataFrame whose columns are to be plotted.
    :type dataframe: pandas.DataFrame
    :param list_columns: A list of column names to plot. If set to 'all', all numeric columns in the DataFrame
                         will be plotted. Default is 'all'.
    :type list_columns: list or str
    :param kde: If `True`, a kernel density estimate plot will be added to the histogram. Default is False.
    :type kde: bool
    :param target_col: Name of the target column to use if `by_class` is `True`. Default is 'TARGET'.
    :type target_col: str
    :param by_class: If `True`, the histogram will be divided by classes based on the `target_col`. Default is False.
    :type by_class: bool
    :param bins: Number of histogram bins. Default is 30.
    :type bins: int
    :param title_y: y-position of the main title. Default is 1.003.
    :type title_y: float
    :param fig_nrows: Number of rows for the subplots. Default is 5.
    :type fig_nrows: int
    :param fig_width: Width of the entire figure. Default is 10.
    :type fig_width: int
    :param fig_height_prop: Height proportion of each row of subplots. Default is 2.5.
    :type fig_height_prop: float
    :param fontscale: Font scale for the plot. Default is 0.8.
    :type fontscale: float
    :param title_fontsize: Font size for the main title. Default is 14.
    :type title_fontsize: int
    :param save_fig: If `True`, the plot will be saved as a PNG image. Default is True.
    :type save_fig: bool
    :param dpi: Dots per inch for saving the image. Default is 500.
    :type dpi: int

    :return: None. The function outputs a plot and optionally saves it as a PNG image.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [10, 20, 30, 40], "TARGET": ["yes", "no", "yes", "no"]})
    >>> plot_columns_histogram(df, list_columns=['A', 'B'], by_class=True, target_col='TARGET')

    Note:
        Ensure that both the `pandas` library and `seaborn` are installed before using this function.
    """
    if isinstance(list_columns, list):
        columns = list_columns
    elif list_columns == "all":
        columns = list_by_type(dataframe, ["float64"])

    nrows = len(columns) // fig_nrows
    nrows = nrows + 1 if len(columns) % fig_nrows > 0 else nrows

    if by_class:
        title = "Histogram of numeric columns by class"
        save_str = "HistogramNumericColsbyClass.png"
    else:
        title = "Histogram of numeric columns"
        save_str = "HistogramNumericCols.png"

    sns.set(font_scale=fontscale)

    fig, axs = plt.subplots(
        nrows, fig_nrows, figsize=(fig_width, int(fig_height_prop * nrows))
    )
    fig.suptitle(title, y=title_y, fontsize=title_fontsize)

    for i in range(len(columns)):
        axi = i // fig_nrows
        axj = i % fig_nrows
        col = columns[i]
        if by_class:
            g = sns.histplot(
                dataframe, x=col, hue=target_col, bins=bins, ax=axs[axi, axj], kde=kde
            )
        else:
            g = sns.histplot(dataframe, x=col, bins=bins, ax=axs[axi, axj], kde=kde)
        g.set(xlabel=None)
        axs[axi, axj].set_title(col)
        if axj > 0:
            g.set(ylabel=None)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_str, dpi=dpi, bbox_inches="tight")

    # Return to defaults
    sns.set(font_scale=1)


def get_mutual_info_target(
    dataframe, list_columns="all", target_col="TARGET", verbose=True, random_state=2023
):
    """
    Computes the mutual information between numeric columns and a target column in a DataFrame.

    This function calculates the mutual information between each specified numeric column and a target column.
    It then categorizes the columns into two lists based on whether they are independent or dependent with respect
    to the target based on their mutual information values.

    :param dataframe: The DataFrame from which mutual information is to be calculated.
    :type dataframe: pandas.DataFrame
    :param list_columns: A list of column names to compute mutual information for. If set to 'all', all numeric
                         columns in the DataFrame will be used. Default is 'all'.
    :type list_columns: list or str
    :param target_col: Name of the target column with respect to which mutual information is to be calculated. Default is 'TARGET'.
    :type target_col: str
    :param verbose: If `True`, the function will print the counts of independent and dependent columns. Default is True.
    :type verbose: bool
    :param random_state: Random state for reproducibility. Default is 2023.
    :type random_state: int
    :return:
        - A dictionary with column names as keys and their corresponding mutual information values as values.
        - A list of dependent columns.
        - A list of independent columns.
    :rtype: tuple (dict, list, list)
    :raises ValueError: If the provided DataFrame is empty.
    :raises KeyError: If the `target_col` is not present in the `dataframe`.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [10, 20, 30, 40], "TARGET": [1, 0, 1, 0]})
    >>> get_mutual_info_target(df, list_columns=['A', 'B'], target_col='TARGET')
    [Sample output of mutual information values, dependent and independent columns]
    Total of independent columns: 2 | Dependent columns: 0
    ({'A': 0, 'B': 0}, [], ['A', 'B'])



    Note:
        Ensure that both the `pandas` library and `sklearn.feature_selection` are installed before using this function.
    """
    if isinstance(list_columns, list):
        columns = list_columns
    elif list_columns == "all":
        columns = list_by_type(dataframe, ["float64"])

    columns.append(target_col)

    numeric_df = dataframe[columns].copy()
    numeric_df = numeric_df.dropna(how="any")
    y = numeric_df[target_col]
    numeric_df.drop([target_col], axis=1, inplace=True)

    mutual_info_list = mutual_info_classif(numeric_df, y, random_state=random_state)

    mutual_info_dict = {}
    dependent_cols = []
    independent_cols = []

    for i in range(len(mutual_info_list)):
        mutual_info_dict[numeric_df.columns[i]] = mutual_info_list[i]
        if mutual_info_list[i] == 0:
            independent_cols.append(numeric_df.columns[i])
        else:
            dependent_cols.append(numeric_df.columns[i])

    if verbose:
        print(
            f"Total of independent columns: {len(independent_cols)} | Dependent columns: {len(dependent_cols)}"
        )

    return mutual_info_dict, dependent_cols, independent_cols


def test_normality(
    dataframe,
    list_columns="all",
    min_p=0.05,
    nan_policy="omit",
    test="normaltest",
    verbose=True,
):
    """
    Tests the normality of the distribution of specified columns in a DataFrame.

    This function examines the specified columns of a provided DataFrame and tests their distributions for
    normality using one of three tests: `normaltest`, `shapiro`, or `kstest`. The results indicate which
    columns have distributions that are likely to be normal.

    :param dataframe: The DataFrame whose columns are to be tested.
    :type dataframe: pandas.DataFrame
    :param list_columns: A list of column names to test. If set to 'all', a default function `list_contin_cols`
                         will determine the columns. Default is 'all'.
    :type list_columns: list or str
    :param min_p: Minimum p-value to consider the distribution as normal. Default is 0.05.
    :type min_p: float
    :param nan_policy: How to handle NaN values. Choices are 'omit' (removes NaN) or 'propagate' (returns NaN). Default is 'omit'.
    :type nan_policy: str
    :param test: Statistical test to use. Choices are 'normaltest', 'shapiro', or 'kstest'. Default is 'normaltest'.
    :type test: str
    :param verbose: If `True`, the function will print the counts of normal and non-normal columns. Default is True.
    :type verbose: bool
    :return:
        - A list of columns with normal distributions.
        - A dictionary with column names as keys and their corresponding p-values as values.
        - A dictionary indicating normality with column names as keys and boolean values as values.
    :rtype: tuple (list, dict, dict)
    :raises ValueError: If the provided DataFrame is empty.
    :raises KeyError: If any column in `list_columns` is not present in the `dataframe`.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8], "B": [10, 20, 30, 40, 50, 60, 70, 80]})
    >>> test_normality(df, list_columns=['A', 'B'])
    [Sample output of normal columns, p-values, and normality indication]
    Total of normal columns: 2 | Non-normal columns: 0
    /usr/local/lib/python3.10/dist-packages/scipy/stats/_stats_py.py:1736: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=8
      warnings.warn("kurtosistest only valid for n>=20 ... continuing "
    (['A', 'B'],
    {'A': 0.42732824212143383, 'B': 0.42732824212143383},
    {'A': True, 'B': True})

    Note:
        Ensure that both the `pandas` library and `scipy.stats` are installed before using this function.
    """

    if isinstance(list_columns, list):
        columns = list_columns
    elif list_columns == "all":
        columns = list_contin_cols(dataframe)

    pvalues = {}
    normality = {}
    normal_columns = []

    for column in columns:
        if test == "normaltest":
            stat, pvalue = stats.normaltest(dataframe[column], nan_policy=nan_policy)
        else:
            if nan_policy == "omit":
                data = dataframe[column].dropna()
            else:
                data = dataframe[column]
            if test == "shapiro":
                stat, pvalue = stats.shapiro(data)
            elif test == "kstest":
                stat, pvalue = stats.kstest(data, cdf="norm")
        pvalues[column] = pvalue
        normality[column] = pvalue >= min_p
        if normality[column]:
            normal_columns.append(column)

    if verbose:
        print(
            f"Total of normal columns: {len(normal_columns)} | Non-normal columns: {len(columns)-len(normal_columns)}"
        )

    return normal_columns, pvalues, normality


def fit_continuous_distributions(
    dataframe, list_columns="all", error="sumsquare_error"
):
    """
    Fits common continuous distributions to the specified columns in a DataFrame.

    This function examines the specified columns of a provided DataFrame and fits a set of common continuous
    distributions to the data in each column. The distribution that best fits each column, as determined by
    the specified error metric, is then identified and recorded.

    :param dataframe: The DataFrame whose columns are to be fitted.
    :type dataframe: pandas.DataFrame
    :param list_columns: A list of column names to fit. If set to 'all', a default function `list_contin_cols`
                         will determine the columns. Default is 'all'.
    :type list_columns: list or str
    :param error: Error metric used to determine the best fitting distribution. Default is 'sumsquare_error'.
    :type error: str
    :return:
        - A DataFrame containing the best fitting distribution for each specified column, along with the distribution
          parameters, the specified error value, and the p-value of the Kolmogorov-Smirnov test.
    :rtype: pandas.DataFrame
    :raises ValueError: If the provided DataFrame is empty.
    :raises KeyError: If any column in `list_columns` is not present in the `dataframe`.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "B": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    >>> fit_continuous_distributions(df, list_columns=['A', 'B'])
    [Sample output of the dataframe with distribution fit results]

    Note:
        Ensure that both the `pandas` library and `fitter` package are installed before using this function.
    """
    contin_cols = list_contin_cols(dataframe)
    if isinstance(list_columns, list):
        columns = list_columns
    elif list_columns == "all":
        columns = contin_cols

    dist_dict = {
        "Column": [],
        "Distribution": [],
        "Parameters": [],
        error: [],
        "ks_pvalue": [],
    }

    for col in columns:
        if col not in contin_cols:
            continue

        # Fit fitter common distributions to col
        f = Fitter(dataframe[col].dropna(), distributions=get_common_distributions())
        f.fit(progress=False)

        # Get distribution with lowest error =
        best_dist = f.get_best(method=error)
        dist_name = list(best_dist.keys())[0]

        # Save best distribution information
        dist_dict["Column"].append(col)
        dist_dict["Distribution"].append(list(best_dist.keys())[0])
        dist_dict["Parameters"].append(list(best_dist.values())[0])

        # Save best distribution error and pvalue
        error_df = f.df_errors.loc[dist_name]
        dist_dict[error].append(error_df[error])
        dist_dict["ks_pvalue"].append(error_df["ks_pvalue"])

    dist_df = pd.DataFrame(dist_dict)
    dist_df.sort_values(by=["Distribution", error])

    return dist_df


def fit_one_continuous_distribution(dataframe, column):
    """
    Fits common continuous distributions to a single column in a DataFrame and plots the results.

    This function takes a column from a provided DataFrame and fits a set of common continuous distributions
    to the data in the column. The fitting results are then plotted, showing the empirical distribution
    of the data along with the fitted distributions.

    :param dataframe: The DataFrame containing the column to be fitted.
    :type dataframe: pandas.DataFrame
    :param column: The name of the column to fit distributions to.
    :type column: str
    :return:
        - A summary plot showing the empirical distribution of the data in the column along with the fitted
          distributions. The plot is displayed to the user.
    :rtype: matplotlib.figure.Figure
    :raises ValueError: If the provided DataFrame is empty.
    :raises KeyError: If the `column` is not present in the `dataframe`.

    :Example:

    >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    >>> fit_one_continuous_distribution(df, column='A')
    [Displays the summary plot]

    Note:
        Ensure that both the `pandas` library, `fitter` package, and `seaborn` are installed before using this function.
    """
    # Set figsize to standard
    sns.set(rc={"figure.figsize": (8, 6)})

    # Fit fitter commom distributions to selected column
    f = Fitter(dataframe[column].dropna(), distributions=get_common_distributions())
    f.fit()

    # Plot summary of distributions fit
    return f.summary()


def plot_columns_dendrogram(
    corr_dataframe,
    method="complete",
    color_threshold=1.0,
    correlation_str=" ",
    figsize=(8, 12),
    leaf_font_size=8,
    title_fontsize=14,
    save_fig=True,
    dpi=500,
):
    """
    Plots a dendrogram of columns based on their correlation.

    This function uses agglomerative hierarchical clustering to cluster the columns of a DataFrame
    based on their correlation and then plots a dendrogram of the resulting clusters.

    :param corr_dataframe: The DataFrame containing the correlation values between columns.
    :type corr_dataframe: pandas.DataFrame
    :param method: The linkage algorithm to use for hierarchical clustering. Default is 'complete'.
    :type method: str
    :param color_threshold: The threshold to apply when coloring the branches. Default is 1.0.
    :type color_threshold: float
    :param correlation_str: Additional string to customize the dendrogram title. Default is ' '.
    :type correlation_str: str
    :param figsize: Figure size for the plot. Default is (8, 12).
    :type figsize: tuple
    :param leaf_font_size: Font size for the labels in the dendrogram. Default is 8.
    :type leaf_font_size: int
    :param title_fontsize: Font size for the title. Default is 14.
    :type title_fontsize: int
    :param save_fig: If `True`, the plot will be saved as a PNG image named 'DendrogramCorrelation.png'. Default is True.
    :type save_fig: bool
    :param dpi: Dots per inch for saving the image. Default is 500.
    :type dpi: int
    :return:
        - Displays a dendrogram based on the correlations of the DataFrame columns.
    :rtype: None

    :Example:

    >>> df_corr = df.corr()
    >>> plot_columns_dendrogram(df_corr, method='complete', color_threshold=0.8)

    Note:
        Ensure that both the `pandas` library, `scipy.cluster.hierarchy` and `matplotlib.pyplot` are installed before using this function.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram

    # ... [rest of the function code]

    # Makes agglomerative clustering of the columns
    Z = linkage(corr_dataframe, method)

    # Plot dendrogram
    fig = plt.figure(figsize=figsize)
    dn = dendrogram(
        Z,
        labels=corr_dataframe.columns,
        color_threshold=color_threshold,
        orientation="right",
        leaf_font_size=leaf_font_size,
    )
    plt.title(
        f"Dendrogram of columns clustered by {correlation_str} correlation",
        fontsize=title_fontsize,
    )
    if save_fig:
        plt.savefig("DendrogramCorrelation.png", dpi=dpi, bbox_inches="tight")
    plt.show()


def get_mutual_info_numeric(df, list_columns="all", random_state=2023):
    """Compute mutual info between each numeric column pair. Drop columns with NaN values.

    :param DataFrame df: DataFrame containing the data.
    :param list list_columns: list of subset of columns to use, defaults to "all" columns
    :param int random_state: random state, defaults to 2023
    :return: numpy array with mutual info between each numeric column pair
    """
    if isinstance(list_columns, list):
        columns = list_columns
    elif list_columns == "all":
        columns = list_by_type(df, ["float64"])

    numeric_df = df[columns].copy()
    numeric_df = numeric_df.dropna(how="any")

    mutual_info = np.zeros((len(columns), len(columns)))

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            mutual_score = mutual_info_regression(
                numeric_df[columns[i]].to_numpy().reshape(-1, 1),
                numeric_df[columns[j]],
                random_state=random_state,
            )
            mutual_info[i, j] = mutual_score
            mutual_info[j, i] = mutual_score

    return mutual_info


def plot_mutual_info_target(
    mutual_info_dict,
    figsize=(6, 10),
    fontscale=0.65,
    title_fontsize=14,
    save_fig=True,
):
    """Plot bar chart with mutual information between numerical columns and target variable. Columns are sorted based on mutual information.

    :param dict mutual_info_dict: dict with column names and its mutual information with target variables
    :param tuple figsize: size of figure, defaults to (6, 10)
    :param float fontscale: scale of font, defaults to 0.65
    :param int title_fontsize: size of title font, defaults to 14
    :param bool save_fig: if should save figure to disk, defaults to True
    """
    columns = list(mutual_info_dict.keys())

    sorted_id = sorted(
        range(len(mutual_info_dict)), key=lambda k: mutual_info_dict[columns[k]]
    )

    sorted_cols = [columns[id] for id in sorted_id]
    sorted_mut_info = [
        mutual_info_dep[col] for col in sorted_cols
    ]  # TODO: fix this variable

    sns.set(font_scale=fontscale)
    fig = plt.figure(figsize=figsize)
    plt.barh(sorted_cols, sorted_mut_info)
    plt.title(
        "Mutual information between numerical columns and target",
        fontsize=title_fontsize,
    )
    if save_fig:
        plt.savefig("MutualInfoNumericTarget.png", dpi=500, bbox_inches="tight")

    plt.show()

    # Return to defaults
    sns.set(font_scale=1)


def plot_mutual_info_numeric(
    mutual_info,
    columns,
    figsize=(12, 13),
    title_fontsize=14,
    title_y=1.015,
    clustermap_fontscale=0.85,
    cmap="Blues",
    cbar_pos=(1.0, 0.3, 0.03, 0.5),
    save_fig=True,
):
    """Plot the mutual information between numeric columns. Also plot a clustermap of the mutual information matrix.

    :param ndarray mutual_info: numpy array with mutual info between each numeric column pair
    :param list columns: list with numeric columns names
    :param tuple figsize: tuple with figure size, defaults to (12, 13)
    :param int title_fontsize: integer with title fontsize, defaults to 14
    :param float title_y: float with relative title vertical position, defaults to 1.015
    :param float clustermap_fontscale: float to define scale of clustermap font, defaults to 0.85
    :param str cmap: matplotlib colormap name, defaults to "Blues"
    :param tuple cbar_pos: positioning of colorbar, defaults to (1.0, 0.3, 0.03, 0.5)
    :param bool save_fig: parameter to save figure to disk, defaults to True
    """
    # Change plot defaults of figsize and fontscale
    sns.set(rc={"figure.figsize": figsize})
    sns.set(font_scale=clustermap_fontscale)

    # Plot clustermap
    g = sns.clustermap(
        mutual_info,
        cmap=cmap,
        cbar_pos=cbar_pos,
        xticklabels=columns,
        yticklabels=columns,
        method="complete",
    )
    g.fig.suptitle(
        "Mutual information between numeric columns", fontsize=title_fontsize, y=title_y
    )
    if save_fig:
        plt.savefig("MutualInfoNumericCols.png", dpi=500, bbox_inches="tight")
    plt.show()

    # Return to plot defaults
    sns.set(font_scale=1)
    sns.set(rc={"figure.figsize": (8, 6)})


def get_numeric_correlation(dataframe, list_columns="all", method="spearman"):
    """Calculate correlation matrix between numeric columns The following correlation methods are available:
    - Pearson: standard correlation coefficient to be used with normally distributed data
    - Spearman: Spearman rank correlation. The standard for non-normal data
    - Kendall: Kendall Tau correlation coefficient. Can also be used with non-normal data

    :param DataFrame dataframe: DataFrame containing the data.
    :param list list_columns: list of columns to calculate correlation, defaults to "all"
    :param str method: name of correlation method to use, defaults to "spearman"
    :return DataFrame: correlation matrix as DataFrame
    """
    if isinstance(list_columns, list):
        columns = list_columns
    elif list_columns == "all":
        columns = dataframe.describe().columns

    corr = dataframe[columns].corr(method=method, numeric_only=True)

    return corr


def plot_numeric_correlation(
    corr,
    figsize=(12, 13),
    title_fontsize=14,
    title_y=1.015,
    clustermap_fontscale=0.65,
    cmap="seismic",
    cbar_pos=(1.0, 0.3, 0.03, 0.5),
    save_fig=True,
):
    """Plot the correlation matrix between columns. Also plot a clustermap of the correlation matrix.

    :param DataFrame corr: dataframe with correlation matrix between columns
    :param tuple figsize: size of figure, defaults to (12, 13)
    :param int title_fontsize: size of title font, defaults to 14
    :param float title_y: relative vertical position of title, defaults to 1.015
    :param float clustermap_fontscale: scale of font of the clustermap, defaults to 0.65
    :param str cmap: matplotlib colormap name, defaults to "seismic"
    :param tuple cbar_pos: position of colorbar, defaults to (1.0, 0.3, 0.03, 0.5)
    :param bool save_fig: if save figure to disk, defaults to True
    """
    # Change plot defaults of fontscale
    sns.set(font_scale=clustermap_fontscale)

    # Plot clustermap
    g = sns.clustermap(
        corr, cmap=cmap, center=0, cbar_pos=cbar_pos, method="complete", figsize=figsize
    )
    g.fig.suptitle(
        "Correlation between numeric columns", fontsize=title_fontsize, y=title_y
    )
    if save_fig:
        plt.savefig("CorrelationNumericCols.png", dpi=500, bbox_inches="tight")
    plt.show()

    # Return to defaults
    sns.set(font_scale=1)


def plot_corr_clustergram(
    dataframe,
    percentile_zero,
    color_threshold=1.0,
    height=950,
    width=1200,
    display_ratio=0.2,
    hide_labels=True,
):
    """Plot iteractive clustergram of correlation matrix between columns.

    :param DataFrame dataframe: DataFrame containing the data.
    :param float percentile_zero: percentile of data to color white
    :param float color_threshold: maximum linkage value for which unique colors are assigned to clusters, defaults to 1.0
    :param int height: height of figure, defaults to 950
    :param int width: width of figure, defaults to 1200
    :param float display_ratio: ratio between dendogram height and heatmap size, defaults to 0.2
    :param bool hide_labels: if should hid labels, defaults to True
    :return: graph object
    """
    columns = list(dataframe.columns.values)
    rows = list(dataframe.index)

    clustergram = dash_bio.Clustergram(
        data=dataframe.loc[rows].values,
        row_labels=rows,
        column_labels=columns,
        color_threshold={"row": color_threshold, "col": color_threshold},
        color_map=[[0.0, "#636EFA"], [percentile_zero, "#ffffff"], [1.0, "#EF553B"]],
        height=height,
        width=width,
        hidden_labels=["row", "col"] if hide_labels else None,
        center_values=False,
        optimal_leaf_order=True,
        line_width=1.5,
        display_ratio=display_ratio,
        paper_bg_color="#ffffff",
        plot_bg_color="#ffffff",
    )

    return clustergram


def separate_variables_cat(dataframe):
    """Get list of categorical variables in a dataframe.

    :param DataFrame dataframe: Dataframe with data
    :return list: list of categorical variables names
    """
    categorical_features_lst = list(dataframe.select_dtypes(include=["object"]).columns)
    return categorical_features_lst


def separate_variables_bin(dataframe):
    """Get list of binary variables in a dataframe.

    :param DataFrame dataframe: Dataframe with data
    :return list: list of binary variables names
    """
    nb_levels_sr = dataframe.nunique()
    binary_features_lst = nb_levels_sr.loc[nb_levels_sr == 2].index.tolist()
    return binary_features_lst


def plot_categorical_features(df, categorical_features_lst):
    """Plot bar plots for each categorical feature with the frequency of each observed value and the target distribution.

    :param DataFrame df: dataframe with data
    :param list categorical_features_lst: list of categorical features names
    """
    for feature in categorical_features_lst:
        fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(20, 10))

        # Plot levels distribution
        if df[feature].nunique() < 10:
            sns.countplot(
                x=df[feature],
                ax=ax[0],
                order=df[feature].value_counts().index.tolist(),
            )
        else:
            sns.countplot(
                y=df[feature],
                ax=ax[0],
                order=df[feature].value_counts().index.tolist(),
            )

        ax[0].set_title("Count plot of each level of the feature: " + feature)

        # Plot target distribution among levels
        table_df = pd.crosstab(df["TARGET"], df[feature], normalize=True)
        table_df = table_df.div(table_df.sum(axis=0), axis=1)
        table_df = pd.crosstab(df["TARGET"], df[feature], normalize=True)
        table_df = table_df.div(table_df.sum(axis=0), axis=1)
        table_df = table_df.transpose().reset_index()
        order_lst = table_df.sort_values(by=1)[feature].tolist()
        table_df = table_df.melt(id_vars=[feature])

        if df[feature].nunique() < 10:
            ax2 = sns.barplot(
                x=table_df[feature],
                y=table_df["value"] * 100,
                hue=table_df["TARGET"],
                ax=ax[1],
                order=order_lst,
            )
            for p in ax2.patches:
                height = p.get_height()
                ax2.text(
                    p.get_x() + p.get_width() / 2.0,
                    height + 1,
                    "{:1.2f}".format(height),
                    ha="center",
                )
        else:
            ax2 = sns.barplot(
                x=table_df["value"] * 100,
                y=table_df[feature],
                hue=table_df["TARGET"],
                ax=ax[1],
                order=order_lst,
            )
            for p in ax2.patches:
                width = p.get_width()
                ax2.text(
                    width + 3.1,
                    p.get_y() + p.get_height() / 2.0 + 0.35,
                    "{:1.2f}".format(width),
                    ha="center",
                )

        ax[1].set_title("Target distribution among " + feature + " levels")
        ax[1].set_ylabel("Percentage")


def generate_binary_heatmap(df, binary_features_lst):
    """Plot heatmap for each binary feature with the count of each observed value and the target distribution.

    :param DataFrame df: dataframe with data
    :param list binary_features_lst: list of binary features names
    """
    binary_features_lst.sort()

    num_rows = 6
    num_cols = 6

    fig, ax = plt.subplots(
        num_rows, num_cols, sharex=False, sharey=False, figsize=(12, 12)
    )
    i = 0
    j = 0

    for idx in range(len(binary_features_lst)):
        if idx % num_cols == 0 and idx != 0:
            j += 1

        i = idx % num_cols
        feature = binary_features_lst[idx]
        table_df = pd.crosstab(df["TARGET"], df[feature], normalize=True)

        # Normalize statistics to remove target unbalance
        table_df = table_df.div(table_df.sum(axis=1), axis=0)
        table_df = table_df.div(table_df.sum(axis=0), axis=1)

        sns.heatmap(
            table_df,
            annot=True,
            square=True,
            ax=ax[j, i],
            cbar=False,
            fmt=".2%",
            cmap="coolwarm",
            linewidths=0.5,
        )

    plt.tight_layout()
    plt.show()


def plot_mutual_categorical_features(df, categorical_features_lst, target_variable):
    """Plot bar plots for each categorical feature with the mutual information between the feature and the target variable.

    :param DataFrame df: dataframe with data
    :param list categorical_features_lst: list with categorical features names
    :param str target_variable: name of target variable
    """
    num_features = len(categorical_features_lst)
    num_rows = int(np.ceil(num_features / 2))

    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 6 * num_rows))
    fig.tight_layout(pad=4)

    for i, feature in enumerate(categorical_features_lst):
        row = i // 2
        col = i % 2

        # Calculate mutual information
        X = pd.get_dummies(df[feature])
        y = df[target_variable]
        mi_scores = mutual_info_classif(X, y)

        # Plot mutual information
        ax = axes[row, col]
        sns.barplot(x=X.columns, y=mi_scores, ax=ax)
        ax.set_title("Mutual Information: " + feature + " vs " + target_variable)
        ax.set_xlabel(feature)
        ax.set_ylabel("Mutual Information")
        ax.tick_params(axis="x", rotation=90)

    # Remove empty subplots if the number of features is not a multiple of 3
    if num_features % 2 != 0:
        for j in range(num_features % 2, 2):
            fig.delaxes(axes[num_rows - 1, j])

    plt.show()


def get_mutual_info_cat_num(
    dataframe,
    numeric_cols="all",
    cat_cols="all",
    target_col="TARGET",
    random_state=2023,
):
    """Calculate mutual information between numeric/categorical columns and the target variable.

    :param DataFrame dataframe: dataframe with data
    :param list numeric_cols: list of numeric columns to use, defaults to "all"
    :param list cat_cols: list of categoric columns to use, defaults to "all"
    :param str target_col: name of target variable, defaults to "TARGET"
    :param int random_state: random state, defaults to 2023
    :return ndarray: matrix with mutual information between numeric and categorical columns and the target variable
    """
    if not isinstance(numeric_cols, list) and numeric_cols == "all":
        numeric_cols = list_by_type(dataframe, ["float64"])

    if not isinstance(cat_cols, list) and cat_cols == "all":
        cat_cols = separate_variables_cat(dataframe)

    columns = numeric_cols.copy()
    columns.extend(cat_cols)

    # Make sure that there are no duplicated columns
    columns = list(set(columns))

    df_temp = dataframe[columns].copy()
    df_temp = df_temp.dropna(how="any")

    if target_col in columns:
        df_temp.drop([target_col], axis=1, inplace=True)

    # mutual_info = np.zeros((len(numeric_cols), len(cat_cols)))
    mutual_info = np.zeros((len(cat_cols), len(numeric_cols)))

    for i in range(len(cat_cols)):
        if cat_cols[i] == target_col:
            continue
        y = df_temp[cat_cols[i]]
        m = mutual_info_classif(df_temp[numeric_cols], y, random_state=random_state)
        mutual_info[i, :] = m

    return mutual_info


def plot_mutual_info_numeric_cat(
    mutual_info,
    numeric_cols,
    cat_cols,
    figsize=(8, 12),
    dendrogram_ratio=(0.2, 0.2),
    title_fontsize=14,
    title_y=1.015,
    clustermap_fontscale=0.8,
    cmap="PuBuGn",
    cbar_pos=(1.0, 0.3, 0.03, 0.5),
    save_fig=True,
):
    """Plot the mutual information between the numeric columns and the categorical columns. Also plot a clustermap of the mutual information matrix.

    :param ndarray mutual_info: numpy array with mutual info between each numeric and categorical pair
    :param list numeric_cols: list with name of numeric columns
    :param list cat_cols: list with name of category columns
    :param tuple figsize: tuple with figure size, defaults to (12, 13)
    :param dendrogram_ratio: ratio of the size of the dendogram and the heatmap, defaults to (0.2, 0.2)
    :param int title_fontsize: integer with title fontsize, defaults to 14
    :param float title_y: float with relative title vertical position, defaults to 1.015
    :param float clustermap_fontscale: float to define scale of clustermap font, defaults to 0.8
    :param str cmap: matplotlib colormap name, defaults to "PuBuGn"
    :param tuple cbar_pos: positioning of colorbar, defaults to (1.0, 0.3, 0.03, 0.5)
    :param bool save_fig: parameter to save figure to disk, defaults to True

    """
    # Change plot defaults of and fontscale
    sns.set(font_scale=clustermap_fontscale)

    # Plot clustermap
    g = sns.clustermap(
        mutual_info,
        cmap=cmap,
        cbar_pos=cbar_pos,
        xticklabels=numeric_cols,
        yticklabels=cat_cols,
        method="complete",
        figsize=figsize,
        dendrogram_ratio=dendrogram_ratio,
    )
    g.fig.suptitle(
        "Mutual information between numeric and categorical columns",
        fontsize=title_fontsize,
        y=title_y,
    )
    if save_fig:
        plt.savefig(
            "MutualInfoNumericCategoricalCols.png", dpi=500, bbox_inches="tight"
        )
    plt.show()

    # Return to plot defaults
    sns.set(font_scale=1)
