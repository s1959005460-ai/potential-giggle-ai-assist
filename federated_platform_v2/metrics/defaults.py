from core.registry import register_metric
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error

@register_metric('accuracy')
def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, (y_pred>0.5).astype(int)))

@register_metric('f1')
def f1(y_true, y_pred):
    return float(f1_score(y_true, (y_pred>0.5).astype(int)))

@register_metric('roc_auc')
def roc_auc(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))

@register_metric('mse')
def mse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred))
