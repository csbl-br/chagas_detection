#! /bin/python3

#%%
# Imports
import os
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier #, BaggingClassifier

from my_tools import fit_models, get_predictions, get_metrics, calculate_performance
from my_tools import plot_roc_curve

nCPU = os.cpu_count() - 1
seed = 42
select_features = False

#%%
# Dataset
dataset = pd.read_csv("./features_data.csv")
images_info = pd.read_csv("./images_info.csv")

# %%
# Train / test split (image level)
train_images = images_info.filename[images_info.set == 'train'].to_list()
train_set = dataset[dataset.filename.isin(train_images)]
x_train = train_set.loc[:, "area":"curvature_energy"]
y_train = train_set["structure"]

test_images = images_info.filename[images_info.set == 'test'].to_list()
test_set = dataset[dataset.filename.isin(test_images)]
x_test = test_set.loc[:, "area":"curvature_energy"]
y_test = test_set["structure"]

#%%
# Train / test split (feature level)
# x = dataset.loc[:, "area":"curvature_energy"]
# y = dataset["structure"]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

#%%
# Scale and select features
if select_features == True:

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    pca = PCA(n_components=0.95)
    pca.fit(x_train_scaled)
    x_train = pca.fit_transform(x_train_scaled)
    x_test = pca.fit_transform(x_test_scaled)

# %%
# Classifiers
rnd_clf = RandomForestClassifier(n_jobs = nCPU, random_state = seed)
svm_clf = SVC(kernel = "rbf", probability = True, random_state = seed)
knn_clf = KNeighborsClassifier(n_jobs = nCPU)

voting_clf = VotingClassifier(
    estimators=[('svc', svm_clf), ('knn', knn_clf), ('rf', rnd_clf)],
    voting = 'soft',
    n_jobs = nCPU)

estimators = [svm_clf, knn_clf, rnd_clf, voting_clf]

#%%
# Fit the parameters
fit_models(x_train, y_train, estimators)

#%%
# Run on train set
# clf = estimators[0]
for clf in estimators:
    ## Get the predictions
    y_pred, y_proba = get_predictions(x_train, clf)

    ## Get the metrics
    fpr, tpr, conf_matrix, _ = get_metrics(y_train, y_pred, y_proba)

    ## Claculate performance
    training_results = calculate_performance(clf, conf_matrix, fpr, tpr)

    ## Print results
    # print(training_results)
    print("Model:", training_results["Model"])
    print("Sensitivity:", round(training_results["Sensitivity"], 4))
    print("Specificity:", round(training_results["Specificity"], 4))
    print("Precision:", round(training_results["Precision"], 4))
    print("Accuracy:", round(training_results["Accuracy"], 4))
    print("F1-score:", round(training_results["F1-score"], 4))
    print("AUC:", round(training_results["AUC"], 4))
    print("\n")
    print(conf_matrix)
    print("\n")
    plot_roc_curve(training_results["FPR"], training_results["TPR"], clf)

    #save_json(training_results, "../results/feature_classification/%s_training.json" % training_results['Model'])

#%%
for clf in estimators:
    # Run on test set
    ## Get the predictions
    y_pred, y_proba = get_predictions(x_test, clf)

    ## Get the metrics
    fpr, tpr, conf_matrix, _ = get_metrics(y_test, y_pred, y_proba)

    ## Calculate performance
    testing_results = calculate_performance(clf, conf_matrix, fpr, tpr)

    ## Print results
    # print(training_results)
    print("Model:", testing_results["Model"])
    print("Sensitivity:", round(testing_results["Sensitivity"], 4))
    print("Specificity:", round(testing_results["Specificity"], 4))
    print("Precision:", round(testing_results["Precision"], 4))
    print("Accuracy:", round(testing_results["Accuracy"], 4))
    print("F1-score:", round(testing_results["F1-score"], 4))
    print("AUC:", round(testing_results["AUC"], 4))
    print("\n")
    print(conf_matrix)
    print("\n")
    plot_roc_curve(testing_results["FPR"], testing_results["TPR"], clf)

    # save_json(testing_results, "../results/feature_classification/%s_testing.json" % testing_results['Model'])

# %%
# Save the models
# from my_tools import save_model
# save_model(rnd_clf, os.path.join("../results/models/rnd_clf.pickle"))
# %%
