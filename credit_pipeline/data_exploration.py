import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process_missing_values(df, limit_percentage, statistic):
    """
    Processes missing values in a DataFrame.
    Arguments:

    df: DataFrame - The DataFrame containing the data.
    limit_percentage: float - The limit (in percentage) for the proportion of missing values.
    statistic: str - The statistic to be used to replace the missing values.
    It can be 'mean', 'median', 'mode', or 'exclude'.
    Returns:
    DataFrame - The DataFrame with the processed missing values.
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
            if statistic == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif statistic == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif statistic == 'mode':
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif statistic == 'exclude':
                df = df.dropna(subset=[column])

    return df

def detect_multicollinearity(dataframe, threshold=0.7, plot=False):
    """
    Identifies multicollinearity in a DataFrame based on a threshold (default: 0.7).
    Multicollinearity occurs when two predictor variables have a high correlation.
    Identifying which variables have this issue helps in the analysis of variable selection
    and further treatment (such as removal, standardization, or dimensionality reduction).

    Arguments:
    - dataframe: DataFrame - The DataFrame containing the data.
    - threshold: float, optional - The correlation threshold to identify multicollinearity.
                                   Default is 0.7.
    - plot: bool, optional - Specifies whether to plot the correlation matrix.
                             Default is False.

    Returns:
    list - A list of tuples containing the pairs of columns with multicollinearity
           and their corresponding correlation values.
    """

    

    # Calculate the correlation matrix
    correlation_matrix = dataframe.corr().abs()

    # Get the list of columns with high correlation (greater than or equal to the threshold)
    collinear_columns = []

    # Iterate over the correlation matrix
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] >= threshold:
                # Add the pair of columns with high correlation to the list
                column_i = correlation_matrix.columns[i]
                column_j = correlation_matrix.columns[j]
                correlation_value = correlation_matrix.iloc[i, j]
                collinear_columns.append((column_i, column_j, correlation_value))

    # Print the pairs of columns with multicollinearity and their correlation values
    for pair in collinear_columns:
        print(f"Multicollinearity detected between: {pair[0]} and {pair[1]}. Correlation: {pair[2]}")

    # Plot the correlation matrix if requested
    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    return collinear_columns


def detect_outliers(dataframe, plot=False, show_top=False):
    """
    Detects outliers in all columns of a DataFrame using the interquartile range (IQR) method.
    Outliers are values that fall below the lower bound or above the upper bound defined by the IQR.
    The function provides an option to plot boxplots for each column.

    Arguments:
    - dataframe: DataFrame - The DataFrame containing the data.
    - plot: bool, optional - Specifies whether to plot boxplots for each column.
                            Default is False.
    - show_top: bool, optional - Specifies whether to sort columns in descending order based on the number of outliers.
                                Default is False.

    Returns:
    dict - A dictionary containing the number of outliers and the proportion of outliers
           for each column.
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
        outlier_mask = (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
        num_outliers = outlier_mask.sum()
        outlier_proportion = num_outliers / len(dataframe)

        # Store the results
        outliers[column] = {'Num_Outliers': num_outliers, 'Proportion': outlier_proportion}

        # Plot the boxplot, if requested
        if plot:
            plt.figure()
            dataframe.boxplot(column=column)
            plt.title(f'Boxplot of {column}')
            plt.show()

    # Sort columns in descending order based on the number of outliers, if show_top is True
    if show_top:
        outliers = dict(sorted(outliers.items(), key=lambda x: x[1]['Num_Outliers'], reverse=True))

    # Print the results
    for column, values in outliers.items():
        print(f"Column: {column}")
        print(f"Number of outliers: {values['Num_Outliers']}")
        print(f"Proportion of outliers: {values['Proportion']}")
        print()

    return outliers
