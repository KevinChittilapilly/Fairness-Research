import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn import linear_model
import sklearn.preprocessing as preprocessing
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
from scipy.stats import entropy, kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocessData(df,isCleanData=False):
    # Define a dictionary where the keys are column names and the values are the mappings for that column
    replacements = {
        'Credit-history': {'A30': 1, 'A31': 2, 'A32': 3, 'A33': 4, 'A34': 5},
        'Purpose': {'A40': 1, 'A41': 2, 'A42': 3, 'A43': 4, 'A44': 5, 'A45': 6, 'A46': 7, 'A47': 8, 'A48': 9, 'A49': 10, 'A410': 11},
        'Savings-account': {'A61': 1, 'A62': 2, 'A63': 3, 'A64': 4, 'A65': 5},
        'Present-employment': {'A71': 1, 'A72': 2, 'A73': 3, 'A74': 4, 'A75': 5},
        'Other-debtors': {'A101': 1, 'A102': 2, 'A103': 3},
        'Property': {'A121': 1, 'A122': 2, 'A123': 3, 'A124': 4},
        'Other-installment': {'A141': 1, 'A142': 2, 'A143': 3},
        'Housing': {'A151': 1, 'A152': 2, 'A153': 3},
        'Telephone': {'A191': 1, 'A192': 2},
        'Foreign-worker': {'A201': 1, 'A202': 2},
        'Status': {'A11': 1, 'A12': 2, 'A13': 3, 'A14': 4},
        'Skill-level': {'A171': 1, 'A172': 2, 'A173': 3, 'A174': 4},
        'Checking-Account': {'A11': 1, 'A12': 2, 'A13': 3, 'A14': 4},
        'Status/sex': {'A91': 1,
                       'A92':  2,
                       'A93':  3,
                       'A94':  4,
                       'A95':  5},
        'Job': {'A171' : 1,
	      'A172' :  2,
        'A173' : 3,
	      'A174' :  4,},

    }

    # Iterate over the dictionary and perform the replacements
    for column, mapping in replacements.items():
        if column in df.columns:
            df[column] = df[column].replace(mapping)
    if ('Credit-amount' in df.columns):
        df['Credit-amount'] = pd.qcut(df['Credit-amount'], q=4,labels=[1,2,3,4])
    if ('Age' in df.columns and not isCleanData):
        df['Age'] = np.where(df['Age'] < 25, 1, 2)
    return df

def getParityforClasses (df):
    col = ""
    if 'Status/sex' in df.columns:
        col = 'Status/sex'
    else: col = 'Sex'
    for i in range(1,7):
        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        y_pred_prob = knn.predict_proba(X_test)

        first_values = [row[0] for row in y_pred_prob]
        second_values = [row[1] for row in y_pred_prob]
        X_test['prob_1'] = first_values
        X_test['prob_2'] = second_values

        sum_male_divorced= X_test.loc[X_test[col] == 1 & (X_test['prob_1'] > 0.5), 'prob_1'].mean()
        sum_male_single= X_test.loc[X_test[col] == 3 & (X_test['prob_1'] > 0.5), 'prob_1'].mean()
        sum_male_married= X_test.loc[X_test[col] == 4 & (X_test['prob_1'] > 0.5), 'prob_1'].mean()
        sum_female_divorced= X_test.loc[X_test[col] == 2 & (X_test['prob_1'] > 0.5), 'prob_1'].mean()
        sum_female_single= X_test.loc[X_test[col] == 5 & (X_test['prob_1'] > 0.5), 'prob_1'].mean()

        print(sum_male_divorced,sum_male_single, sum_male_married,sum_female_divorced,sum_female_single)

        sum_male_divorced= X_test.loc[X_test[col] == 1 & (X_test['prob_2'] > 0.5), 'prob_2'].mean()
        sum_male_single= X_test.loc[X_test[col] == 3 & (X_test['prob_2'] > 0.5), 'prob_2'].mean()
        sum_male_married= X_test.loc[X_test[col] == 4 & (X_test['prob_2'] > 0.5), 'prob_2'].mean()
        sum_female_divorced= X_test.loc[X_test[col] == 2 & (X_test['prob_2'] > 0.5), 'prob_2'].mean()
        sum_female_single= X_test.loc[X_test[col] == 5 & (X_test['prob_2'] > 0.5), 'prob_2'].mean()

        print(sum_male_divorced,sum_male_single, sum_male_married,sum_female_divorced,sum_female_single)
        print("----------------------------------------")


def calculateIndependence(df):
   

    columns_of_interest = df.columns.drop(['target'])


    chi_squared_results = []

    for column in columns_of_interest:
        contingency_table = pd.crosstab(df['target'], df[column])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi_squared_results.append({'Attribute': column, 'Chi-Square': chi2, 'P-Value': p})

    results_df = pd.DataFrame(chi_squared_results)

    results_df.sort_values(by='P-Value', ascending=True, inplace=True)

    print("Associations between 'target' and other columns:")
    print(results_df)
    return results_df

def eclat(prefix, items, min_support, frequent_itemsets, support_count):
    while items:
        i, itids = items.pop()
        itids_sup = len(itids)
        if itids_sup >= min_support:
            new_prefix = prefix + [i]
            frequent_itemsets.append((new_prefix, itids_sup))
            support_count[frozenset(new_prefix)] = itids_sup
            suffix = []
            for j, ojtids in items:
                jtids = itids & ojtids
                if len(jtids) >= min_support:
                    suffix.append((j, jtids))
            eclat(new_prefix, suffix, min_support, frequent_itemsets, support_count)

def run_eclat(transactions, min_support_percentage, targets):
    min_support = len(transactions) * min_support_percentage
    data = {}
    for tid, t in enumerate(transactions):
        for item in t:
            data.setdefault(item, set()).add(tid)

    items = [(item, tidset) for item, tidset in data.items()]
    frequent_itemsets = []
    support_count = {}
    eclat([], items, min_support, frequent_itemsets, support_count)

    # Filter frequent itemsets to only include those containing any of the targets
    target_frequent_itemsets = [itemset for itemset in frequent_itemsets if any(target in itemset[0] for target in targets)]

    return target_frequent_itemsets, support_count



def getTransactions (df):
    transactions = []
    print("hello")
    for index, row in df.iterrows():
        transaction = []
        for col, val in row.items():
            item = f"{col}={val}"
            transaction.append(item)
        transactions.append(transaction)
    return transactions

def calculate_confidence(frequent_itemsets, support_count, targets):
    rules = []
    for itemset, support in frequent_itemsets:
        for target in targets:
            # Check if the target is in the itemset
            if target in itemset:
                # Create the antecedent by removing the target from the itemset
                antecedent = frozenset(itemset) - frozenset([target])
                consequent = frozenset([target])

                if antecedent:  # Ensure antecedent is not empty
                    antecedent_support = support_count[antecedent]
                    confidence = support / antecedent_support
                    rules.append((antecedent, consequent, support, confidence))
    rules.sort(key=lambda rule: rule[3], reverse=True)
    return rules

def calculateParityusingKNN(df):
    for i in range(1,7):
        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        y_pred_prob = knn.predict_proba(X_test)

        first_values = [row[0] for row in y_pred_prob]
        second_values = [row[1] for row in y_pred_prob]
        X_test['prob_1'] = first_values
        X_test['prob_2'] = second_values

        age_below_thes= X_test.loc[X_test['Age'] == 1 , 'prob_1'].mean()

        print(age_below_thes)

        age_above_thres= X_test.loc[X_test['Age'] == 2, 'prob_1'].mean()

        print(age_above_thres)
        print("----------------------------------------")

def calculateParityusingLR(df):
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_pred_prob = logreg.predict_proba(X_test)

    first_values = [row[0] for row in y_pred_prob]
    second_values = [row[1] for row in y_pred_prob]
    X_test['prob_1'] = first_values
    X_test['prob_2'] = second_values

    age_below_thes= X_test.loc[X_test['Age'] == 1 , 'prob_1'].mean()

    print(age_below_thes)

    age_above_thres= X_test.loc[X_test['Age'] == 2, 'prob_1'].mean()

    print(age_above_thres)
    print("----------------------------------------")
    return abs(age_above_thres-age_below_thes)


def createSamples(df):
    sampled_dfs = []
    num_samples = 50
    seed_value = 42
    for i in range(num_samples):
        sampled_df = df.sample(n=300, random_state=seed_value + i)
        sampled_dfs.append(sampled_df)
    return sampled_dfs
    
def getStatsfromData(sampled_dfs):
    id=1
    parity = []
    age_below_thes=[]
    age_above_thres=[]
    prop_above_thres=[]
    prop_below_thes=[]
    for sample_df in sampled_dfs:
        X = sample_df.drop('target', axis=1)
        y = sample_df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        y_pred_prob = logreg.predict_proba(X_test)

        first_values = [row[0] for row in y_pred_prob]
        second_values = [row[1] for row in y_pred_prob]
        X_test['prob_1'] = first_values
        X_test['prob_2'] = second_values
        age_below_thes.append(X_test.loc[X_test['Age'] == 1 , 'prob_1'].mean())
        # print(age_below_thes)

        age_above_thres.append(X_test.loc[X_test['Age'] == 2, 'prob_1'].mean())
        parity_diff = abs(X_test.loc[X_test['Age'] == 2, 'prob_1'].mean() - X_test.loc[X_test['Age'] == 1, 'prob_1'].mean())
        print(parity_diff)
        parity.append(parity_diff)
        id+=1

        total_age = sample_df['Age'].value_counts()[1] + sample_df['Age'].value_counts()[2]
        prop_above_thres.append( sample_df['Age'].value_counts()[2]/total_age)
        prop_below_thes.append(sample_df['Age'].value_counts()[1]/total_age)
        filtered_rows = sample_df[(sample_df['Age'] == 1) & (sample_df['target'] == 1)]
        total_rows = len(filtered_rows)
        filtered_rows = sample_df[(sample_df['Age'] == 2) & (sample_df['target'] == 1)]
        total_rows = len(filtered_rows)
        return [age_below_thes,
                age_above_thres,prop_above_thres,
                prop_below_thes,
                total_rows]
def compute_advanced_metafeatures(dataframe, numeric_cols, categorical_cols):
    
    metafeatures = {}

    # Extract specified numeric and categorical columns
    numeric_data = dataframe[numeric_cols]
    categorical_data = dataframe[categorical_cols]

    # Basic Metafeatures
    metafeatures['num_instances'] = len(dataframe)
    metafeatures['num_features'] = len(numeric_cols) + len(categorical_cols)
    metafeatures['num_missing_values'] = dataframe.isnull().sum().sum()
    metafeatures['num_numeric_features'] = numeric_data.shape[1]
    metafeatures['num_categorical_features'] = categorical_data.shape[1]

    # Numerical Features' Metafeatures
    if not numeric_data.empty:
        imputer = SimpleImputer(strategy='mean')
        numeric_cols_imputed = imputer.fit_transform(numeric_data)
        metafeatures['mean_skewness'] = skew(numeric_cols_imputed).mean()
        metafeatures['mean_kurtosis'] = kurtosis(numeric_cols_imputed).mean()

        # PCA Components for 95% variance
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_cols_imputed)
        pca = PCA(n_components=0.95)
        pca.fit(numeric_scaled)
        metafeatures['pca_components_95_var'] = pca.n_components_

    # Categorical Features' Metafeatures
    if not categorical_data.empty:
        metafeatures['min_categories'] = categorical_data.apply(lambda x: x.nunique()).min()
        metafeatures['max_categories'] = categorical_data.apply(lambda x: x.nunique()).max()
        metafeatures['mean_categories'] = categorical_data.apply(lambda x: x.nunique()).mean()

        # Entropy of categorical features
        entropy_values = []
        for col in categorical_data:
            value_counts = categorical_data[col].value_counts(normalize=True, dropna=True)
            entropy_values.append(entropy(value_counts))
        metafeatures['mean_entropy'] = np.mean(entropy_values)

    return metafeatures   
def createMetaFeaturesDataframe(dfs):
    metafeatures_df = pd.DataFrame()
    metafeatures_list = []
    for df in dfs:
        df = df.drop('target',axis=1)
        numeric_columns = ['Duration']
        categorical_columns = ['Checking-Account','Credit-history', 'Purpose',
            'Credit-amount', 'Savings-account', 'Present-employment',
            'Installment-rate', 'Status/sex', 'Other-debtors', 'Present-residence',
            'Property', 'Age', 'Other-installment', 'Housing', 'Existing-credits',
            'Job', 'liable', 'Telephone', 'Foreign-worker']
        metafeatures = compute_advanced_metafeatures(df, numeric_columns, categorical_columns)
        metafeatures_list.append(metafeatures)
    metafeatures_df = pd.DataFrame(metafeatures_list)
    chi_squared_results = []
    for sample_df in dfs:
        contingency_table = pd.crosstab(sample_df['target'],sample_df['Age'])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi_squared_results.append(chi2)
    metafeatures_df['correlation'] = chi_squared_results
    parity_values = []
    for df in dfs:
        parity_values.append(calculateParityusingLR(df))
    metafeatures_df['target'] = parity_values
    metafeatures_df.fillna(metafeatures_df.mean(), inplace=True)
    return metafeatures_df
