# importing required libraries
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
from task_generate import gen_default_tokernizer, simpleReasoning
from utils import TrainingInfo

def get_loss(model, criterion, src):
    output = model(src)
    vocab_size = output.size(-1)
    loss = criterion(
        output[:, :-1].contiguous().view(-1, vocab_size),
        src[:, 1:].contiguous().view(-1),
    )
    return loss

def get_mask(start_ind_list: List, max_len: int) -> torch.Tensor:
    n = len(start_ind_list)
    mask = torch.ones((n, max_len), dtype=torch.long)  # Initialize with ones
    for i, l in enumerate(start_ind_list):
        mask[i, :l] = 0  # Set the first L[i] elements to zero in each row
    return mask

@torch.no_grad()
def loss_err(model, criterion, src, mask):
    """
    Calculates the loss and err of prediction by model on prompts
    """
    model.eval()
    output = model(src)
    vocab_size = output.size(-1)
    loss = criterion(
        output[:, :-1].contiguous().view(-1, vocab_size),
        src[:, 1:].contiguous().view(-1),
    )
    tmp = output.argmax(dim=2)[:, :-1] == src[:, 1:]
    err = 1 - torch.sum(tmp.cpu() * mask[:, :-1], dtype=torch.float) / torch.sum(mask)
    return loss, err

def make_scheduler(optimizer, config):
    if config.schedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif config.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.num_epoch
        )
    return scheduler

#######################################################################################
############################    Main training function    #############################
#######################################################################################
def train_fresh_sample(
    model,
    config,
    optimizer,
    scheduler,
):
    
    num_epoch = config.num_epoch
    vocab = torch.arange(config.vocab_size).type(torch.LongTensor)
    criterion = (
            nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
            if config.label_smoothing > 0
            else nn.CrossEntropyLoss()
        )

    tokenizer = gen_default_tokernizer()
    assert len(tokenizer["map"]) == config.vocab_size, "Tokenizer does not have correct vocab_size"
    config.vocab_size = len(tokenizer["map"])
    task = simpleReasoning(tokenizer, max_variables=config.max_variables, max_parenthesis=config.max_parenthesis, max_seq_len=config.max_seq_len)
    src_test, src_test_info = task.formatted_sample(num_steps=config.test_sample_size)
    src_test = torch.tensor(np.stack(src_test), dtype=torch.long, device=config.device)
    mask_test = get_mask(src_test_info["solution_start_ind"], config.max_seq_len)

    training_info = TrainingInfo()
    for epoch in tqdm(range(num_epoch)):
        model.train()
        optimizer.zero_grad()
        src, src_info = task.formatted_sample(num_steps=config.batch_size)
        src = torch.tensor(np.stack(src), dtype=torch.long, device=config.device)
        mask = get_mask(src_info["solution_start_ind"], config.max_seq_len)

        loss = get_loss(model, criterion, src)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()  # useful if dropout or batchnorm etc is turned on
            loss_train, train_err = loss_err(model, criterion, src, mask)
            loss_test, test_err = loss_err(model, criterion, src_test, mask_test)

        training_info.add_epoch_data(
            epoch=epoch,
            loss=[loss_train.item(), loss_test.item()],
            error=[train_err.item(), test_err.item()],
            batch={"batch_size": src.size(0), "learning_rate": optimizer.param_groups[0]['lr']}
        )

        scheduler.step()

        if epoch % config.measurements_every_epoch == 0 or epoch <= config.measurements_initial_few_epoch:
            pass ### TO DO: add analysis, measurements, and plots later

            if config.print_output:
                val1, val2 = training_info.losses[epoch][1], training_info.errors[epoch][1]
                print(
                    f"----> Epoch: {epoch+1:>5}, Test Loss: {val1:.3f}, Test Error: {val2:.3f}"
                )

        if (1 + epoch) % (config.num_epoch // config.n_save) == 0 or (
            config.up_to_first_save
            and (1 + epoch)
            in [
                np.power(2, k)
                for k in range(int(np.log2(config.num_epoch // config.n_save)))
            ]
        ):
            #out_path = os.path.join(config.out_dir, f"ckpt_{epoch + 1}.pt")
            #torch.save(model.state_dict(), out_path)
            pass ### TO DO: save checkpoints later

    return model, training_info
    