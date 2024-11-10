from typing import Dict
from utils import set_parameters, collect_fidelity
from fedtabdiff_modules import init_model, generate_samples, decode_samples
import flwr as fl
from sdv.metadata import SingleTableMetadata
from logging import INFO, DEBUG
from flwr.common.logger import log

def get_evaluate_server_fn(test_loader, exp_params):
    """The evaluate function for the server. It will be executed by Flower after every round.

    Args:
        test_loader (torch.utils.data.DataLoader): test data loader
        exp_params (dict): experiment parameters

    Returns:
        function: evaluate function
    """
    def evaluate_server(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
    ):
        """Evaluate the centralized model (server) on the locally held test set.

        Args:
            server_round (int): server iteration round 
            parameters (fl.common.NDArrays): model parameters
            config (Dict[str, fl.common.Scalar]): configuration dictionary

        Returns:
            dict: Server-side fidelity score
        """
        fidelity_score = {}
        # evaluate server every eval_rate_server rounds
        if (server_round % exp_params['eval_rate_server'] == 0) and (server_round > 0):

            # initialize model
            synthesizer, diffuser = init_model(exp_params=exp_params)
            synthesizer = synthesizer.to(exp_params['device'])

            # Update model with the latest parameters
            log(INFO, f"[Server evaluation, server round: {server_round}")
            set_parameters(synthesizer, parameters)  
            
            # get test set and label
            # print(f"Loading eval set")
            test_set, test_label = test_loader
            
            # generate new samples
            generated_samples = generate_samples(
                synthesizer=synthesizer,
                diffuser=diffuser,
                encoded_dim=exp_params['encoded_dim'],
                last_diff_step=exp_params['diffusion_steps'],
                label=test_label
            )
            # decode generated samples, i.e. numeric upscaling + categorical inverse encoding
            generated_samples_df = decode_samples(
                samples=generated_samples,
                cat_dim=exp_params['cat_dim'],
                n_cat_emb=exp_params['n_cat_emb'],
                num_attrs=exp_params['num_attrs'],
                cat_attrs=exp_params['cat_attrs'],
                num_scaler=exp_params['num_scaler'],
                vocab_per_attr=exp_params['vocab_per_attr'],
                label_encoder=exp_params['label_encoder'],
                embeddings=synthesizer.get_embeddings()
            )

            # initialize svd metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=test_set)

            # collect fidelity score
            fidelity_score = collect_fidelity(
                real_data=test_set,
                synthetic_data=generated_samples_df,
                metadata=metadata)

            log(INFO, f"Server-side fidelity {fidelity_score.get('fidelity')}")
        
        return None, fidelity_score

    return evaluate_server
