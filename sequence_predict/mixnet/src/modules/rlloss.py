import torch
import torch.nn.functional as F


# def to_gpu(v):
#     if torch.cuda.is_available():
#         v = v.cuda()
#     return v


def rl_single_label_loss(model_outputs, labels, loss_type, sess_length = None, weight = None):
    """Calculate single label positive loss."""
    if model_outputs.shape[0] == 0 or labels.shape[0] == 0:
        return model_outputs.new_tensor(0.)

    if sess_length is None:
        return model_outputs.new_tensor(0.)

    assert model_outputs.shape[0] == labels.shape[0]

    _, predict_id = torch.max(model_outputs, dim=-1)
    reward = model_outputs.new_zeros(predict_id.shape[0])

    s = 0
    for seg in sess_length:
        if 1 in weight[s:s + seg] or 0 in weight[s:s + seg]:
            r = 0
            seg_loss_weight_arr = weight[s:s + seg]
            seg_question_id_arr = labels[s:s + seg]
            seg_predict_id = predict_id[s:s + seg]

            seg_loss_weight_arr_list = seg_loss_weight_arr.cpu().numpy().tolist()
            if 1 in weight[s: s + seg]:
                index_1 = seg_loss_weight_arr_list.index(1)
                positive_id = seg_question_id_arr[index_1]
            else:
                #index_1 = -1
                positive_id = -1

            if 0 in weight[s: s + seg]:
                index_0 = seg_loss_weight_arr_list.index(0)
                negative_id = seg_question_id_arr[index_0]
            else:
                #index_0 = -1
                negative_id = -1

            positive_id = model_outputs.new_tensor(positive_id, dtype=torch.long)
            negative_id = model_outputs.new_tensor(negative_id, dtype=torch.long)
            for sub_seg in range(seg):
                if seg_predict_id[sub_seg] == positive_id:
                    r += 400
                    # print("True")
                elif seg_predict_id[sub_seg] != negative_id and negative_id != -1:
                    # r+=1
                    r += 0
                else:
                    # r-=1
                    r -= 100
                reward[s + sub_seg] = r
        s += seg

    if loss_type == 'cross_entropy':
        loss_list = F.cross_entropy(model_outputs, predict_id, reduce=False)
        # loss = torch.mean(reward * loss_list)
        loss = torch.sum(reward * loss_list)
        # loss = F.cross_entropy(model_outputs, labels, reduction='sum')
    else:
        raise NotImplementedError(f'The {loss_type} loss is not implemented.')
    return loss
