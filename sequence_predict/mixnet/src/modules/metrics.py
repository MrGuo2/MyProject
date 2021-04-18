from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch


def calc_metrics(predict, label):
    """Calculate single label metrics."""
    if torch.is_tensor(predict):
        predict = predict.cpu().numpy()
    if torch.is_tensor(label):
        label = label.cpu().numpy()

    if label.shape[0] == 0:
        accuracy, precision, recall, f1 = 0., 0., 0., 0.
    else:
        accuracy = accuracy_score(label, predict)
        precision = precision_score(label, predict, average='macro')
        recall = recall_score(label, predict, average='macro')
        f1 = f1_score(label, predict, average='macro')

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return metrics_dict


def calc_multi_label_metrics(predict, label):
    """Calculate multi-label metrics."""
    if label.shape[0] == 0:
        accuracy = 0.
    else:
        true_predictions = torch.eq(predict, label).float()
        label_match_num = torch.sum((true_predictions + label).eq(2).float()).item()
        label_num = torch.sum(label).item()
        accuracy = label_match_num / (label_num + 1e-10)

    metrics_dict = {
        'accuracy': accuracy
    }

    return metrics_dict
