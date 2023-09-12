import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import dash_bio


def process_missing_values(df, limit_percentage, statistic):
    """Processes missing values in a DataFrame.

    :param DataFrame df: The DataFrame containing the data.
    :param float limit_percentage: The limit (in percentage) for the proportion of missing values.
    :param str statistic: The statistic to be used to replace the missing values. It can be 'mean', 'median', 'mode', or 'exclude'.
    :return DataFrame: The DataFrame with the processed missing values.
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


def detect_multicollinearity(dataframe, threshold=0.7, plot=False):
    """Identifies multicollinearity in a DataFrame based on a threshold (default: 0.7). Multicollinearity occurs when two predictor variables have a high correlation.
    Identifying which variables have this issue helps in the analysis of variable selection and further treatment (such as removal, standardization, or dimensionality reduction).

    :param DataFrame dataframe: The DataFrame containing the data.
    :param float threshold:The correlation threshold to identify multicollinearity. Default is 0.7.
    :param bool plot: Specifies whether to plot the correlation matrix. Default is False.
    :return list: A list of tuples containing the pairs of columns with multicollinearity and their corresponding correlation values.
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
    """Detects outliers in all columns of a DataFrame using the interquartile range (IQR) method. Outliers are values that fall below the lower bound or above the upper bound defined by the IQR. The function provides an option to plot boxplots for each column.

    :param DataFrame dataframe: The DataFrame containing the data.
    :param bool plot: Specifies whether to plot boxplots for each column. Default is False.
    :param bool show_top: Specifies whether to sort columns in descending order based on the number of outliers. Default is False.
    :return dict: A dictionary containing the number of outliers and the proportion of outliers for each column.
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
