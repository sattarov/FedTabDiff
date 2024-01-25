import flwr as fl
import os
import pandas as pd
from utils import get_parameters, set_parameters, collect_fidelity
from fedtabdiff_modules import init_model, train_model, generate_samples, decode_samples
from sdv.metadata import SingleTableMetadata


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, cid, train_loader, test_loader, exp_params):
        """Initializes the client.

        Args:
            cid (int): Client ID.
            train_loader (torch.utils.data.DataLoader): train data loader
            test_loader (torch.utils.data.DataLoader): test data loader
            exp_params (dict): experiment parameters
        """
        self.cid = cid
        self.device = exp_params['device']
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.exp_params = exp_params
        self.train_epochs_fn = train_model
        self.synthesizer, self.diffuser = init_model(exp_params=exp_params)

    def get_parameters(self, config):
        """ Get the current model parameters from the server.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            List[np.ndarray]: Model parameters.
        """
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.synthesizer)

    def fit(self, parameters, config):
        """Train the model on the locally held training set.

        Args:
            parameters (dict): Client model parameters.
            config (dict): Configuration dictionary.

        Returns:
            List[np.ndarray]: Updated model parameters.
            int: Number of samples used for training.
            dict: Dictionary containing training loss.
        """
        # server_round = config['server_round']
        n_samples = len(self.train_loader.dataset)
        print(f"[Client {self.cid}] fit, n_samples: {n_samples}")
        set_parameters(self.synthesizer, parameters)
        loss = self.train_epochs_fn(
            synthesizer=self.synthesizer,
            diffuser=self.diffuser,
            train_loader=self.train_loader,
            exp_params=self.exp_params
        )

        return get_parameters(self.synthesizer), n_samples, {'loss': loss}

    def evaluate(self, parameters, config):
        """Evaluate the model on the locally held test set.

        Args:
            parameters (List[np.ndarray]): Client model parameters.
            config (dict): Configuration dictionary.

        Returns:
            float: Loss on the test set.
            int: Number of samples in the test set.
            dict: Dictionary containing evaluation metric result.
        """

        # get server round
        server_round = config['server_round']
        # get number of samples
        n_samples = len(self.test_loader[0])
        # initialize fidelity score
        fidelity_score = {}
        loss = 0.0

        # evaluate server every eval_rate_client rounds
        if (server_round % config['eval_rate_client'] == 0) and (server_round > 0):
            print(f"[Client {self.cid}] evaluate, server round: {server_round}, n_samples: {n_samples}")
            # initialize model
            set_parameters(self.synthesizer, parameters)

            # get test set and label
            test_set, test_label = self.test_loader
            # generate new samples
            generated_samples = generate_samples(
                synthesizer=self.synthesizer,
                diffuser=self.diffuser,
                encoded_dim=self.exp_params['encoded_dim'],
                last_diff_step=self.exp_params['diffusion_steps'],
                label=test_label
            )
            # decode generated samples, i.e. numeric upscaling + categorical inverse encoding
            generated_samples_df = decode_samples(
                samples=generated_samples,
                cat_dim=self.exp_params['cat_dim'],
                n_cat_emb=self.exp_params['n_cat_emb'],
                num_attrs=self.exp_params['num_attrs'],
                cat_attrs=self.exp_params['cat_attrs'],
                num_scaler=self.exp_params['num_scaler'],
                vocab_per_attr=self.exp_params['vocab_per_attr'],
                label_encoder=self.exp_params['label_encoder'],
                embeddings=self.synthesizer.get_embeddings()
            )

            # initialize svd metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=test_set)

            # evaluate metrics, fidelity
            fidelity_score = collect_fidelity(
                real_data=test_set,
                synthetic_data=generated_samples_df,
                metadata=metadata)

        print(f"Client-side fidelity {fidelity_score.get('fidelity')}")
        return loss, n_samples, fidelity_score


def get_client_fn(train_loaders, test_loaders, exp_params):
    """Return a function to construct a client.
    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    
    Args:
        train_loaders (torch.utils.data.DataLoader): train data loader
        exp_params (dict): experiment parameters
    Returns:
        function to construct a client
    """
    def client_fn(cid) -> FlowerClient:
        train_loader = train_loaders[int(cid)]
        test_loader = test_loaders[int(cid)]
        return FlowerClient(cid, train_loader, test_loader, exp_params)

    return client_fn


def get_eval_config(exp_params):
    def client_eval_config(server_round: int):
        """Return training configuration dict for each round."""
        client_config = {
            "server_round": server_round,
            **exp_params
        }
        return client_config

    return client_eval_config
