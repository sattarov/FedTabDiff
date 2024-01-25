import numpy as np
from collections import OrderedDict
from typing import List
import torch
from sdmetrics.single_column import KSComplement, TVComplement


def get_parameters(net) -> List[np.ndarray]:
    """ Get the parameters of a model.

    Args:
        net (torch model): torch model

    Returns:
        List[np.ndarray]: extracted model parameters
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """ Set the parameters of a model.

    Args:
        net (torch model): torch model
        parameters (List[np.ndarray]): model parameters
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def collect_fidelity(real_data, synthetic_data, metadata):
    """ Collect column fidelity scores.
        TVComplement used for categorical columns (https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/tvcomplement)
        KSComplement used for numerical columns (https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/kscomplement)
        
    Args:
        real_data (pd.DataFrame): original data
        synthetic_data (pd.DataFrame): synthetic data
        metadata (sdv object): sdv metadata

    Returns:
        dict: column fidelity scores
    """
    fidelity = {}

    for col_name, col_meta in metadata.columns.items():
        if col_meta['sdtype'] == 'categorical':
            score = TVComplement.compute(real_data=real_data[col_name], synthetic_data=synthetic_data[col_name])
        elif col_meta['sdtype'] == 'numerical':
            score = KSComplement.compute(real_data=real_data[col_name], synthetic_data=synthetic_data[col_name])
        else:
            raise Exception('undefined column type')

        fidelity[col_name] = score
    fidelity_score = {'fidelity': np.mean(list(fidelity.values()))}

    return fidelity_score
