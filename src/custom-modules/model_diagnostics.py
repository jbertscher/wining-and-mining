from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns


# Model diagnostics
def model_results(y, y_pred, y_pred_proba):
    print(classification_report(y, y_pred))
    print('Area under ROC:', roc_auc_score(y, y_pred_proba[:, 1]))
    plot_cm(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    
# Cross-validation model fitting and diagnostics
def model_results_cv(X, y, classifier):
    y_pred = cross_val_predict(classifier, X, y, cv = 10)
    y_pred_proba = cross_val_predict(classifier, X, y, cv = 10, method='predict_proba')
    model_results(y, y_pred, y_pred_proba)


# Plot a confusion matrix
def plot_cm(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    fig = plt.figure(figsize = (5,5))
    sns.set(font_scale = 1.25)
    sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = ',.0f', annot_kws = {'size': 14})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show();