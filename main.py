import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np

sns.set_context("talk", font_scale=1.5)

# #Load the data from the Excel file
# df = pd.read_excel('indic-stat-circonscriptions-legislatives-2022.xlsx')
#
# ###############################
# #      DATA CLEANING          #
# ###############################
#
# # Drop the first 7 rows of the dataframe
# df = df.iloc[6:]
# # Set the columns labels and the index of the dataframe
# df.columns = df.iloc[0]
# df = df.iloc[1:]
# df.set_index('circo', inplace=True)
# # Drop all columns labeled as selected in the dataframe
# df.drop(columns=["N°Panneau"], inplace=True)
# df.drop(columns=["Nom"], inplace=True)
# df.drop(columns=["Prénom"], inplace=True)
# df.drop(columns=["Sexe"], inplace=True)
# df.drop(columns=["Sièges"], inplace=True)
#
# # Get the unique values from the selected column
# unique_values = df.loc[:, 'Nuance'].stack().unique()
# # Print the unique values
# print(unique_values)
#
# # categorize all nuances
# nuances_dict = {
#     "DXG": 'FL',
#     "RDG": 'FL',
#     "NUP": 'L',
#     "ECO": 'L',
#     "DVG": 'L',
#     "ENS": 'C',
#     "DVC": 'C',
#     "LR": 'R',
#     "DVD": 'R',
#     "UDI": 'R',
#     "DSV": 'FR',
#     "REC": 'FR',
#     "RN": 'FR',
#     "DXD": 'FR',
#     "DIV": 'DIV',
#     "REG": 'DIV'
# }
#
# # Find and replace all occurrences of a string in the dataframe
# df.replace(nuances_dict, inplace=True)
#
# # Create a NumPy array with 200 rows and 3 columns of zeros
# FLdf = pd.DataFrame(np.zeros((df.shape[0], 3)))
# FLdf.columns = ["Voix", "% Voix/Ins", "% Voix/Exp"]
# Ldf = pd.DataFrame(np.zeros((df.shape[0], 3)))
# Ldf.columns = ["Voix", "% Voix/Ins", "% Voix/Exp"]
# Cdf = pd.DataFrame(np.zeros((df.shape[0], 3)))
# Cdf.columns = ["Voix", "% Voix/Ins", "% Voix/Exp"]
# Rdf = pd.DataFrame(np.zeros((df.shape[0], 3)))
# Rdf.columns = ["Voix", "% Voix/Ins", "% Voix/Exp"]
# FRdf = pd.DataFrame(np.zeros((df.shape[0], 3)))
# FRdf.columns = ["Voix", "% Voix/Ins", "% Voix/Exp"]
# DIVdf = pd.DataFrame(np.zeros((df.shape[0], 3)))
# DIVdf.columns = ["Voix", "% Voix/Ins", "% Voix/Exp"]
#
# for rrow in range(df.shape[0]):
#     for ccol in range(df.shape[1]):
#         row = rrow - 1
#         col = ccol - 1
#         if df.iat[row, col] == 'FL':
#             FLdf.iat[row, 0] = FLdf.iat[row, 0] + df.iat[row, col + 1]
#             FLdf.iat[row, 1] = FLdf.iat[row, 1] + df.iat[row, col + 2]
#             FLdf.iat[row, 2] = FLdf.iat[row, 2] + df.iat[row, col + 3]
#         if df.iat[row, col] == 'L':
#             Ldf.iat[row, 0] = Ldf.iat[row, 0] + df.iat[row, col + 1]
#             Ldf.iat[row, 1] = Ldf.iat[row, 1] + df.iat[row, col + 2]
#             Ldf.iat[row, 2] = Ldf.iat[row, 2] + df.iat[row, col + 3]
#         if df.iat[row, col] == 'C':
#             Cdf.iat[row, 0] = Cdf.iat[row, 0] + df.iat[row, col + 1]
#             Cdf.iat[row, 1] = Cdf.iat[row, 1] + df.iat[row, col + 2]
#             Cdf.iat[row, 2] = Cdf.iat[row, 2] + df.iat[row, col + 3]
#         if df.iat[row, col] == 'R':
#             Rdf.iat[row, 0] = Rdf.iat[row, 0] + df.iat[row, col + 1]
#             Rdf.iat[row, 1] = Rdf.iat[row, 1] + df.iat[row, col + 2]
#             Rdf.iat[row, 2] = Rdf.iat[row, 2] + df.iat[row, col + 3]
#         if df.iat[row, col] == 'FR':
#             FRdf.iat[row, 0] = FRdf.iat[row, 0] + df.iat[row, col + 1]
#             FRdf.iat[row, 1] = FRdf.iat[row, 1] + df.iat[row, col + 2]
#             FRdf.iat[row, 2] = FRdf.iat[row, 2] + df.iat[row, col + 3]
#         row = rrow - 1
#         col = ccol - 1
#         if df.iat[row, col] == 'DIV':
#             DIVdf.iat[row, 0] = DIVdf.iat[row, 0] + df.iat[row, col + 1]
#             DIVdf.iat[row, 1] = DIVdf.iat[row, 1] + df.iat[row, col + 2]
#             DIVdf.iat[row, 2] = DIVdf.iat[row, 2] + df.iat[row, col + 3]
#
# FLdf.to_csv('fldf.csv')
# Ldf.to_csv('ldf.csv')
# Cdf.to_csv('cdf.csv')
# Rdf.to_csv('rdf.csv')
# FRdf.to_csv('frdf.csv')
# DIVdf.to_csv('divdf.csv')

# Here we read a new dataset made out of the previous work, so we don't have to run everything again

df = pd.read_excel('datafull.xlsx', index_col='circo', header=0)
# Replace the string "nd" with NaN in the dataframe
df = df.replace('nd', np.nan)
df.dropna(inplace=True)
df.describe()

###############################
#    Rescaling the dataset    #
###############################

# Standard if there are outliers, MinMax in general

# scaler = StandardScaler()
scaler = MinMaxScaler()

df[['actdip_PEU_T', 'FL% Voix/Exp_T', 'L% Voix/Exp_T', 'C% Voix/Exp_T', 'R% Voix/Exp_T', 'FR% Voix/Exp_T',
    'DIV% Voix/Exp_T']] = scaler.fit_transform(
    df[['actdip_PEU', 'FL% Voix/Exp', 'L% Voix/Exp', 'C% Voix/Exp', 'R% Voix/Exp', 'FR% Voix/Exp', 'DIV% Voix/Exp']])

###############################
#       Scatter Plot          #
###############################

# Doing some scatter plot to visualize the abstention against some different data
df.plot(kind='scatter', x='actdip_PEU_T', y='FL% Voix/Exp')
df.plot(kind='scatter', x='actdip_PEU_T', y='L% Voix/Exp')
df.plot(kind='scatter', x='actdip_PEU_T', y='C% Voix/Exp')
df.plot(kind='scatter', x='actdip_PEU_T', y='R% Voix/Exp')
df.plot(kind='scatter', x='actdip_PEU', y='FR% Voix/Exp')
df.plot(kind='scatter', x='actdip_PEU', y='DIV% Voix/Exp')


###############################
#          KMeans             #
###############################


# To identify the optimum number of clusters and kmeans for abstention and express vote for far left party
def optimise_and_process_k_means(data, max_k, title):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Cluster Number')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

    for i in range(3, 7, 1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        df['kmeans' + str(i)] = kmeans.labels_
        plt.scatter(x=df['actdip_PEU'], y=df[data.columns[1][:len(data.columns[1]) - 2]], c=df['kmeans' + str(i)])
        plt.title(title + ' ' + str(i))
        plt.show()


optimise_and_process_k_means(df[['actdip_PEU_T', 'FL% Voix/Exp_T']], 10, 'Far Left')
optimise_and_process_k_means(df[['actdip_PEU_T', 'L% Voix/Exp_T']], 10, 'Left')
optimise_and_process_k_means(df[['actdip_PEU_T', 'C% Voix/Exp_T']], 10, 'Center')
optimise_and_process_k_means(df[['actdip_PEU_T', 'R% Voix/Exp_T']], 10, 'Right')
optimise_and_process_k_means(df[['actdip_PEU_T', 'FR% Voix/Exp_T']], 10, 'Far Right')
optimise_and_process_k_means(df[['actdip_PEU_T', 'DIV% Voix/Exp_T']], 10, 'Diverse')


###############################
#             GMM             #
###############################

def gmm_optimize_and_process(data, max_k, title):
    models = []
    for n in range(1, max_k):
        models.append(GaussianMixture(n, covariance_type='full', random_state=0).fit(data))
    gmm_model_comparisons = pd.DataFrame({
        "BIC": [m.bic(data) for m in models],
        "AIC": [m.aic(data) for m in models]})
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=gmm_model_comparisons[["BIC", "AIC"]])
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.show()

    n = input('Insert cluster number according to figure')

    gmm = GaussianMixture(int(n), covariance_type='full', random_state=0).fit(data)
    labels = gmm.predict(data)
    plt.figure(figsize=(9, 7))
    plt.scatter(x=df[data.columns[0][:len(data.columns[0]) - 2]], y=df[data.columns[1][:len(data.columns[1]) - 2]],
                c=labels)
    plt.title(title)
    plt.show()


gmm_optimize_and_process(df[['actdip_PEU_T', 'FL% Voix/Exp_T']], 10, 'Far Left')
gmm_optimize_and_process(df[['actdip_PEU_T', 'L% Voix/Exp_T']], 10, 'Left')
gmm_optimize_and_process(df[['actdip_PEU_T', 'C% Voix/Exp_T']], 10, 'Center')
gmm_optimize_and_process(df[['actdip_PEU_T', 'R% Voix/Exp_T']], 10, 'Right')
gmm_optimize_and_process(df[['actdip_PEU_T', 'FR% Voix/Exp_T']], 10, 'Far Right')
gmm_optimize_and_process(df[['actdip_PEU_T', 'DIV% Voix/Exp_T']], 10, 'Diverse')
