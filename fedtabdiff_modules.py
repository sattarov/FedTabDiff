import pandas as pd
from tqdm import tqdm
# models
from MLPSynthesizer import MLPSynthesizer
from BaseDiffuser import BaseDiffuser
import numpy as np
from torch import optim
import torch
from torch import nn


def init_model(exp_params):
    """ Initialize model

    Args:
        exp_params (dict): experiment parameters

    Returns:
        synthesizer: synthesizer model
        diffuser: diffuser model
    """
    print(f"Initializing FedTabDiff model")
    # define synthesizer
    synthesizer = MLPSynthesizer(
        d_in=exp_params['encoded_dim'],
        hidden_layers=exp_params['mlp_layers'],
        activation=exp_params['activation'],
        n_cat_tokens=exp_params['n_cat_tokens'],
        n_cat_emb=exp_params['n_cat_emb'],
        n_classes=exp_params['n_classes'],
        embedding_learned=False
    )
    # define diffuser
    diffuser = BaseDiffuser(
        total_steps=exp_params['diffusion_steps'],
        beta_start=exp_params['diffusion_beta_start'],
        beta_end=exp_params['diffusion_beta_end'],
        device=exp_params['device'],
        scheduler=exp_params['scheduler'])

    return synthesizer, diffuser


def train_model(synthesizer, diffuser, train_loader, exp_params, optimizer=None):
    """ Training function for FedTabDiff

    Args:
        synthesizer: synthesizer model
        diffuser: diffuser model
        train_loader (torch data loader): training data loader
        exp_params (dict): experiment parameters
        optimizer (torch optimizer, optional): optimizer. Defaults to None.

    Returns:
        float: training loss
    """
    device = exp_params['device']
    client_rounds = exp_params['client_rounds']
    
    # init optimizer
    parameters = filter(lambda p: p.requires_grad, synthesizer.parameters())
    if optimizer is None:
        optimizer = optim.Adam(parameters, lr=exp_params['learning_rate'])
    
    # init loss function
    loss_fnc = nn.MSELoss()
    total_losses = []
    
    # iterate over client rounds
    round = 0

    # iterate over distinct mini-batches
    for _, (batch_cat, batch_num, batch_y) in enumerate(train_loader):
        
        # set network in training mode
        synthesizer.train()
        synthesizer.to(device)

        # move batch to device
        batch_cat = batch_cat.to(device)
        batch_num = batch_num.to(device)
        batch_y = batch_y.to(device)

        # sample timestamps t
        timesteps = diffuser.sample_timesteps(n=batch_cat.shape[0])

        # get cat embeddings
        batch_cat_emb = synthesizer.embed_categorical(x_cat=batch_cat)

        # concat cat & num
        batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)
        
        # add noise
        batch_noise_t, noise_t = diffuser.add_gauss_noise(x_num=batch_cat_num, t=timesteps)
        
        # conduct forward encoder/decoder pass
        predicted_noise = synthesizer(x=batch_noise_t, timesteps=timesteps, label=batch_y)

        # compute train loss
        train_losses = loss_fnc(
            input=noise_t,
            target=predicted_noise,
        )

        # reset encoder and decoder gradients
        optimizer.zero_grad()

        # run error back-propagation
        train_losses.backward()

        # optimize encoder and decoder parameters
        optimizer.step()

        # collect rec error losses
        total_losses.append(train_losses.detach().cpu().numpy())

        round += 1
        if round >= client_rounds:
            break

    # average of rec errors
    total_losses_mean = np.mean(np.array(total_losses))

    return total_losses_mean


@torch.no_grad()
def generate_samples(
        synthesizer,
        diffuser,
        encoded_dim,
        last_diff_step,
        n_samples=None, 
        label=None
    ):
    """ Generation of samples. 
        For unconditional sampling use n_samples, for conditional sampling provide label.

    Args:
        synthesizer (_type_): synthesizer model
        diffuser (_type_): diffuzer model
        encoded_dim (int): transformed data dimension 
        last_diff_step (int): total number of diffusion steps
        n_samples (int, optional): number of samples to sample. Defaults to None.
        label (tensor, optional): list of labels for conditional sampling. Defaults to None.

    Returns:
        torch.Tensor: matrix of generated samples
    """
    device = next(synthesizer.parameters()).device
    if (n_samples is None) and (label is None):
        raise Exception("either n_samples or label needs to be given")

    if label is not None:
        n_samples = len(label)
        
    # initialize noise
    z_norm = torch.randn((n_samples, encoded_dim)).float()

    label = label.to(device)
    z_norm = z_norm.to(device)

    # iterate over diffusion steps
    pbar = tqdm(iterable=reversed(range(0, last_diff_step)))
    for i in pbar:

        # update progress bar
        pbar.set_description(f"SAMPLING STEP: {i:4d}")

        # sample timestamps t
        t = torch.full((n_samples,), i, dtype=torch.long).to(device)

        # conduct forward encoder/decoder pass
        model_out = synthesizer(z_norm, t, label)

        # reverse diffusion step, i.e. noise removal
        z_norm = diffuser.p_sample_gauss(model_out, z_norm, t)

    return z_norm

def decode_samples(
        samples,
        cat_dim,
        n_cat_emb,
        num_attrs,
        cat_attrs,
        num_scaler,
        vocab_per_attr,
        label_encoder,
        embeddings,
    ):
    """ Decoding function for unscaling numeric attributes and inverse encoding of categorical attributes.
        Used once synthetic data is generated. 

    Args:
        sample (tensor): input samples for decoding
        cat_dim (int): categorical dimension
        n_cat_emb (int): size of categorical embeddings
        num_attrs (list): numeric attributes
        cat_attrs (list): categorical attributes
        num_scaler (_type_): numeric scaler from sklearn
        vocab_per_attr (dict): vocabulary of distinct values in attribute
        label_encoder (_type_): categorical encoder
        embeddings (_type_): embeddings

    Returns:
        pandas DataFrame: decoded samples
    """

    # split sample into numeric and categorical parts
    # samples = samples.cpu().numpy()
    samples_num = samples[:, cat_dim:]
    samples_cat = samples[:, :cat_dim]

    # denormalize numeric attributes
    z_norm_upscaled = num_scaler.inverse_transform(samples_num.cpu().numpy())
    z_norm_df = pd.DataFrame(z_norm_upscaled, columns=num_attrs)

    # reshape back to batch_size * n_dim_cat * cat_emb_dim
    samples_cat = samples_cat.reshape(-1, len(cat_attrs), n_cat_emb)

    # compute batch-wise calculation of distances because for datasets with large number of embedding tokens can be memory costly
    batch_size = 2048
    n_samples = len(samples)
    z_cat_df_list = []

    # iterate over generated categorical samples
    for i in range(0, n_samples, batch_size):
        # get batch of samples
        samples_cat_subset = samples_cat[i: i+batch_size]
        
        # compute pairwise distances between embeddings and generated samples
        distances = torch.cdist(x1=embeddings, x2=samples_cat_subset)

        # create temp dataframes for collection of intermediate results
        z_cat_df_temp = pd.DataFrame(index=range(len(samples_cat_subset)), columns=cat_attrs)

        for attr_idx, attr_name in enumerate(cat_attrs):
        
            # get vocab indices for attribute
            attr_emb_idx = list(vocab_per_attr[attr_name])
        
            # get distances for attribute
            attr_distances = distances[:, attr_emb_idx, attr_idx]
        
            # get nearest embedding index
            _, nearest_idx = torch.min(attr_distances, dim=1)
        
            # convert to numpy
            nearest_idx = nearest_idx.cpu().numpy()
        
            # map emb indices back to column indices
            z_cat_df_temp[attr_name] = np.array(attr_emb_idx)[nearest_idx]

        # collect temp DFs            
        z_cat_df_list.append(z_cat_df_temp)

    # concat DFs
    z_cat_df = pd.concat(z_cat_df_list, ignore_index=True)

    # inverse transform categorical attributes
    z_cat_df = z_cat_df.apply(label_encoder.inverse_transform)
    
    # concat numeric and categorical attributes
    sample_decoded = pd.concat([z_cat_df, z_norm_df], axis=1)

    return sample_decoded
