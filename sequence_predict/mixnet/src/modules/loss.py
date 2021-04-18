import torch
import torch.nn.functional as F


def single_label_pos_loss(model_outputs, labels, loss_type):
    """Calculate single label positive loss."""
    if model_outputs.shape[0] == 0 or labels.shape[0] == 0:
        return model_outputs.new_tensor(0.)

    assert model_outputs.shape[0] == labels.shape[0]

    if loss_type == 'cross_entropy':
        loss = F.cross_entropy(model_outputs, labels, reduction='sum')
    else:
        raise NotImplementedError(f'The {loss_type} loss is not implemented.')
    return loss


def single_label_neg_loss(model_outputs, labels, loss_type):
    """Calculate single label negative loss."""
    if model_outputs.shape[0] == 0 or labels.shape[0] == 0:
        return model_outputs.new_tensor(0.)

    assert model_outputs.shape[0] == labels.shape[0]

    softmax_results = F.softmax(model_outputs, dim=-1)
    if loss_type == 'l2':
        logits = torch.gather(softmax_results, dim=1, index=labels.unsqueeze(1)).squeeze(1)
        loss = 0.5 * torch.sum(logits ** 2)
    elif loss_type == 'cross_entropy':
        logits = torch.gather(-torch.log(1 - softmax_results), dim=1, index=labels.unsqueeze(1)).squeeze(1)
        loss = torch.sum(logits)
    else:
        raise NotImplementedError(f'The {loss_type} negative loss is not implemented.')

    return loss


def multi_label_loss(model_outputs, labels, loss_type):
    """Calculate multi-label loss."""
    if model_outputs.shape[0] == 0 or labels.shape[0] == 0:
        return model_outputs.new_tensor(0.)

    if loss_type == 'sigmoid_cross_entropy':
        logits = F.sigmoid(model_outputs)
        loss = F.binary_cross_entropy(logits, labels, reduction='sum')
    else:
        raise NotImplementedError(f'The {loss_type} multi_label loss is not implemented.')

    return loss
