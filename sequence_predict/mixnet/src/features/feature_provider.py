import os

from features.feature_bert import FeatureBert
from features.feature_label import FeatureLabel
from features.feature_multi_label import FeatureMultiLabel
from features.feature_seq import FeatureSeq
from features.feature_text import FeatureText, FeatureMultiTurnText
from features.feature_value import FeatureValue
from features.feature_vector import FeatureVector

def get_feature_handler(feature_type, feature_config, global_config, tokenizer):
    if feature_type == 'label':
        return FeatureLabel(global_config.multi_turn_separator, feature_config.get('mask', '-1'))
    elif feature_type == 'multi_label':
        return FeatureMultiLabel(global_config.multi_label_separator)

    elif feature_type == 'value':
        if 'valid_key_file' in feature_config:
            feature_config['valid_key_file'] = os.path.join(global_config.config_dir, feature_config['valid_key_file'])
        return FeatureValue(feature_config)
    elif feature_type == 'seq':
        return FeatureSeq(feature_config)
    elif feature_type == 'text':
        return FeatureText(feature_config, tokenizer)
    elif feature_type == 'multi_turn_text':
        return FeatureMultiTurnText(feature_config, tokenizer)
    elif feature_type == 'bert':
        return FeatureBert(feature_config)
    elif feature_type == 'vector':
        return FeatureVector(feature_config)

    else:
        raise NotImplementedError(f'Unknown feature type: {feature_type}.')
