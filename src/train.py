# -*- coding: utf-8 -*-
import os
import gc
import logging
import numpy as np
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.losses import SimclrCriterion
from optimisers import get_optimiser
from utils import save_events_table, key_averages_with_stack

def pretrain_train_batch(inputs, encoder, mlp, optimiser, criterion, sample_count, run_loss, profiler=None):
    inputs = inputs.cuda(non_blocking=True)

    # Forward pass
    optimiser.zero_grad()

    # retrieve the 2 views
    x_i, x_j = torch.split(inputs, [3, 3], dim=1)

    # Get the encoder representation
    h_i = encoder(x_i)

    h_j = encoder(x_j)

    # Get the nonlinear transformation of the representation
    z_i = mlp(h_i)

    z_j = mlp(h_j)

    # Calculate NT_Xent loss
    loss = criterion(z_i, z_j)

    loss.backward()

    optimiser.step()

    torch.cuda.synchronize()

    sample_count += inputs.size(0)

    batch_loss = loss.item()

    if profiler is not None:
        profiler.step()
    return batch_loss

def pretrain_train_epoch(encoder, mlp, args, train_dataloader, optimiser, criterion,lr_decay,epoch,profiler=None):
    ''' epoch loop '''

    sample_count = 0
    run_loss = 0

    for i, (inputs, _) in enumerate(train_dataloader):


        run_loss += pretrain_train_batch(inputs, encoder, mlp, optimiser, criterion, sample_count, run_loss, profiler=profiler)

    epoch_pretrain_loss = run_loss / len(train_dataloader)

    ''' Update Schedulers '''
    # TODO: Improve / add lr_scheduler for warmup
    if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
        wu_lr = (float(epoch+1) / args.warmup_epochs) * args.learning_rate
        save_lr = optimiser.param_groups[0]['lr']
        optimiser.param_groups[0]['lr'] = wu_lr
    else:
        # After warmup, decay lr with CosineAnnealingLR
        lr_decay.step()

    ''' Printing '''
    if args.print_progress:  # only validate using process 0
        logging.info('\n[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

        args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
        args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

    state = {
        #'args': args,
        'encoder': encoder.state_dict(),
        'mlp': mlp.state_dict(),
        'optimiser': optimiser.state_dict(),
        'epoch': epoch,
    }

    torch.save(state, args.checkpoint_dir)
    return epoch_pretrain_loss

def pretrain(encoder, mlp, dataloaders, args):
    ''' Pretrain script - SimCLR

        Pretrain the encoder and projection head with a Contrastive NT_Xent Loss.
    '''

    mode = 'pretrain'

    ''' Optimisers '''
    optimiser = get_optimiser((encoder, mlp), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = (1e-12 / args.warmup_epochs) * args.learning_rate

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, args.n_epochs, eta_min=0.0, last_epoch=-1)

    ''' Loss / Criterion '''
    criterion = SimclrCriterion(batch_size=args.batch_size, normalize=True,
                                temperature=args.temperature).cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    patience_counter = 0

    n_batches = len(dataloaders['pretrain'])
    print(f"n_batches: {n_batches}")

    ''' Pretrain loop '''
    for epoch in range(args.n_epochs):

        # Train models
        encoder.train()
        mlp.train()

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['pretrain'])
        else:
            train_dataloader = dataloaders['pretrain']

        if(epoch==args.n_epochs-1):
            with torch.profiler.profile(
                activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True,
                    schedule=torch.profiler.schedule(
                    wait=n_batches-3,
                    warmup=1,
                    active=1,
                    repeat=0)
            ) as p:
                with torch.profiler.record_function("model_pretraining_epoch"):
                
                    epoch_pretrain_loss = pretrain_train_epoch(encoder,mlp,args,train_dataloader,optimiser,criterion,lr_decay,epoch,profiler=p)

                #print(p.function_events)
                table = key_averages_with_stack(p.profiler.function_events).table(
                    sort_by="self_cuda_time_total", row_limit=-1, top_level_events_only=False)
                print(table)

                p.export_stacks(os.path.join(args.model_dir,'profiler_pretrain.stacks'),'self_cuda_time_total')

                #write table to txt file
                with open(os.path.join(args.model_dir, 'profiler_pretrain.txt'), 'w') as profiler_log:
                    profiler_log.write(table)

                #write the profiler output to csv with custom function
                save_events_table(key_averages_with_stack(p.profiler.function_events), os.path.join(args.model_dir, 'profiler_pretrain.csv'),times_path=os.path.join(args.model_dir, 'final_times_pretrain.txt'),row_limit=-1, top_level_events_only=False)

        else:
            epoch_pretrain_loss = pretrain_train_epoch(encoder,mlp,args,train_dataloader,optimiser,criterion,lr_decay,epoch,profiler=None)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_loss = epoch_pretrain_loss

        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                break

        epoch_pretrain_loss = None  # reset loss


    #del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory

def supervised_train_batch(inputs, target, encoder, mlp, optimiser, criterion, sample_count, run_loss, profiler=None):
    inputs = inputs.cuda(non_blocking=True)

    target = target.cuda(non_blocking=True)

    # Forward pass
    optimiser.zero_grad()

    h = encoder(inputs)

    # Take pretrained encoder representations
    output = mlp(h)

    loss = criterion(output, target)

    loss.backward()

    optimiser.step()

    torch.cuda.synchronize()

    sample_count += inputs.size(0)

    batch_loss = loss.item()

    predicted = output.argmax(1)

    acc = (predicted == target).sum().item() / target.size(0)

    batch_top1 = acc

    _, output_topk = output.topk(5, 1, True, True)

    acc_top5 = (output_topk == target.view(-1, 1).expand_as(output_topk)
                ).sum().item() / target.size(0)  # num corrects

    batch_top5 = acc_top5
    if profiler is not None:
        profiler.step()

    return batch_loss, batch_top1, batch_top5

def supervised_train_epoch(encoder, mlp, args, train_dataloader, optimiser, criterion,lr_decay,epoch,profiler=None):
    ''' epoch loop '''
    sample_count = 0
    run_loss = 0
    run_top1 = 0.0
    run_top5 = 0.0

    for i, (inputs, target) in enumerate(train_dataloader):
        batch_loss, batch_top1, batch_top5  = supervised_train_batch(inputs, target, encoder, mlp, optimiser, criterion, sample_count, run_loss, profiler=profiler)
        run_loss += batch_loss
        run_top1 += batch_top1
        run_top5 += batch_top5

    epoch_pretrain_loss = run_loss / len(train_dataloader)  # sample_count
    epoch_pretrain_acc = run_top1 / len(train_dataloader)
    epoch_pretrain_acc_top5 = run_top5 / len(train_dataloader)

    ''' Update Schedulers '''
    # TODO: Improve / add lr_scheduler for warmup
    if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
        wu_lr = (float(epoch+1) / args.warmup_epochs) * args.learning_rate
        save_lr = optimiser.param_groups[0]['lr']
        optimiser.param_groups[0]['lr'] = wu_lr
    else:
        # After warmup, decay lr with CosineAnnealingLR
        lr_decay.step()

    ''' Printing '''
    if args.print_progress:  # only validate using process 0
        logging.info('\n[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

        args.writer.add_scalars('supervised_epoch_acc', {
                                'pretrain': epoch_pretrain_acc}, epoch+1)
        args.writer.add_scalars('supervised_epoch_acc_top5', {
                                'pretrain': epoch_pretrain_acc_top5}, epoch+1)
        args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
        args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

    state = {
        #'args': args,
        'encoder': encoder.state_dict(),
        'mlp': mlp.state_dict(),
        'optimiser': optimiser.state_dict(),
        'epoch': epoch,
    }

    torch.save(state, args.checkpoint_dir)
    return epoch_pretrain_loss

def supervised(encoder, mlp, dataloaders, args):
    ''' Supervised Train script - SimCLR

        Supervised Training encoder and train the supervised classification head with a Cross Entropy Loss.
    '''

    mode = 'pretrain'

    ''' Optimisers '''
    # Only optimise the supervised head
    optimiser = get_optimiser((encoder, mlp), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = (1e-12 / args.warmup_epochs) * args.learning_rate

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, args.n_epochs, eta_min=0.0, last_epoch=-1)

    ''' Loss / Criterion '''
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    patience_counter = 0

    n_batches = len(dataloaders['train'])
    print(f"n_batches: {n_batches}")

    ''' Pretrain loop '''
    for epoch in range(args.n_epochs):

        # Train models
        encoder.train()
        mlp.train()

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['train'])
        else:
            train_dataloader = dataloaders['train']

        if(epoch == args.n_epochs - 1):
            with torch.profiler.profile(
                    activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                        profile_memory=True,
                        with_stack=True,
                        with_flops=True,
                        schedule=torch.profiler.schedule(
                        wait=n_batches-3,
                    warmup=1,
                    active=1,
                    repeat=0)
                ) as p:

                epoch_pretrain_loss = supervised_train_epoch(encoder,mlp,args,train_dataloader,optimiser,criterion,lr_decay,epoch,profiler=p)

                table = key_averages_with_stack(p.profiler.function_events).table(
                    sort_by="self_cuda_time_total", row_limit=-1, top_level_events_only=False)
                print(table)

                p.export_stacks(os.path.join(args.model_dir,'profiler_pretrain.stacks'),'self_cuda_time_total')

                #write table to txt file
                with open(os.path.join(args.model_dir, 'profiler_pretrain_supervised.txt'), 'w') as profiler_log:
                    profiler_log.write(table)

                #write the profiler output to csv with custom function
                save_events_table(key_averages_with_stack(p.profiler.function_events), os.path.join(args.model_dir, 'profiler_pretrain_supervised.csv'),times_path=os.path.join(args.model_dir, 'final_times_pretrain_supervised.txt'),row_limit=-1, top_level_events_only=False)
        else:
            epoch_pretrain_loss = supervised_train_epoch(encoder,mlp,args,train_dataloader,optimiser,criterion,lr_decay,epoch,profiler=None)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_loss = epoch_pretrain_loss

        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                break

        epoch_pretrain_loss = None  # reset loss

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory


def finetune(encoder, mlp, dataloaders, args):
    ''' Finetune script - SimCLR

        Freeze the encoder and train the supervised classification head with a Cross Entropy Loss.
    '''

    mode = 'finetune'

    ''' Optimisers '''
    # Only optimise the supervised head
    optimiser = get_optimiser((mlp,), mode, args)

    ''' Schedulers '''
    # Cosine LR Decay
    lr_decay = lr_scheduler.CosineAnnealingLR(optimiser, args.finetune_epochs)

    ''' Loss / Criterion '''
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    best_valid_acc = 0.0
    patience_counter = 0

    n_batches = len(dataloaders['train'])
    print(f"n_batches: {n_batches}")
    ''' Pretrain loop '''
    for epoch in range(args.finetune_epochs):

        # Freeze the encoder, train classification head
        encoder.eval()
        mlp.train()

        sample_count = 0
        run_loss = 0
        run_top1 = 0.0
        run_top5 = 0.0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.finetune_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['train'])
        else:
            train_dataloader = dataloaders['train']

        ''' epoch loop '''
        for i, (inputs, target) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)

            target = target.cuda(non_blocking=True)

            # Forward pass
            optimiser.zero_grad()

            # Do not compute the gradients for the frozen encoder
            with torch.no_grad():
                h = encoder(inputs)

            # Take pretrained encoder representations
            output = mlp(h)

            loss = criterion(output, target)

            loss.backward()

            optimiser.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()

            predicted = output.argmax(1)

            acc = (predicted == target).sum().item() / target.size(0)

            run_top1 += acc

            _, output_topk = output.topk(5, 1, True, True)

            acc_top5 = (output_topk == target.view(-1, 1).expand_as(output_topk)
                        ).sum().item() / target.size(0)  # num corrects

            run_top5 += acc_top5

        epoch_finetune_loss = run_loss / len(dataloaders['train'])  # sample_count

        epoch_finetune_acc = run_top1 / len(dataloaders['train'])

        epoch_finetune_acc_top5 = run_top5 / len(dataloaders['train'])

        ''' Update Schedulers '''
        # Decay lr with CosineAnnealingLR
        lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Finetune] loss: {:.4f},\t acc: {:.4f}, \t acc_top5: {:.4f}\n'.format(
                epoch_finetune_loss, epoch_finetune_acc, epoch_finetune_acc_top5))

            args.writer.add_scalars('finetune_epoch_loss', {'train': epoch_finetune_loss}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc', {'train': epoch_finetune_acc}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc_top5', {
                                    'train': epoch_finetune_acc_top5}, epoch+1)
            args.writer.add_scalars(
                'finetune_lr', {'train': optimiser.param_groups[0]['lr']}, epoch+1)


        #Log the validation losses.
        valid_loss, valid_acc, valid_acc_top5 = evaluate(
            encoder, mlp, dataloaders, 'valid', epoch, args)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if valid_acc >= best_valid_acc:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_acc = valid_acc

            # saving using process (rank) 0 only as all processes are in sync

            state = {
                #'args': args,
                'encoder': encoder.state_dict(),
                'supp_mlp': mlp.state_dict(),
                'optimiser': optimiser.state_dict(),
                'epoch': epoch
            }

            torch.save(state, (args.checkpoint_dir[:-3] + "_finetune.pt"))
        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                break

        epoch_finetune_loss = None  # reset loss
        epoch_finetune_acc = None
        epoch_finetune_acc_top5 = None
        
    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory


def evaluate(encoder, mlp, dataloaders, mode, epoch, args):
    ''' Evaluate script - SimCLR

        evaluate the encoder and classification head with Cross Entropy loss.
    '''

    epoch_valid_loss = None  # reset loss
    epoch_valid_acc = None  # reset acc
    epoch_valid_acc_top5 = None

    ''' Loss / Criterion '''
    criterion = nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)

    # Evaluate both encoder and class head
    encoder.eval()
    mlp.eval()

    # initilize Variables
    sample_count = 0
    run_loss = 0
    run_top1 = 0.0
    run_top5 = 0.0

    # Print setup for distributed only printing on one node.
    if args.print_progress:
            # tqdm for process (rank) 0 only when using distributed training
        eval_dataloader = tqdm(dataloaders[mode])
    else:
        eval_dataloader = dataloaders[mode]

    ''' epoch loop '''
    for i, (inputs, target) in enumerate(eval_dataloader):

        # Do not compute gradient for encoder and classification head
        encoder.zero_grad()
        mlp.zero_grad()

        inputs = inputs.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)

        # Forward pass

        h = encoder(inputs)

        output = mlp(h)

        loss = criterion(output, target)

        torch.cuda.synchronize()

        sample_count += inputs.size(0)

        run_loss += loss.item()

        predicted = output.argmax(-1)

        acc = (predicted == target).sum().item() / target.size(0)

        run_top1 += acc

        _, output_topk = output.topk(5, 1, True, True)

        acc_top5 = (output_topk == target.view(-1, 1).expand_as(output_topk)
                    ).sum().item() / target.size(0)  # num corrects

        run_top5 += acc_top5

    epoch_valid_loss = run_loss / len(dataloaders[mode])  # sample_count

    epoch_valid_acc = run_top1 / len(dataloaders[mode])

    epoch_valid_acc_top5 = run_top5 / len(dataloaders[mode])

    ''' Printing '''
    if args.print_progress:  # only validate using process 0
        logging.info('\n[{}] loss: {:.4f},\t acc: {:.4f},\t acc_top5: {:.4f} \n'.format(
            mode, epoch_valid_loss, epoch_valid_acc, epoch_valid_acc_top5))

        if mode != 'test':
            args.writer.add_scalars('finetune_epoch_loss', {mode: epoch_valid_loss}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc', {mode: epoch_valid_acc}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc_top5', {
                                    mode: epoch_valid_acc_top5}, epoch+1)

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory

    return epoch_valid_loss, epoch_valid_acc, epoch_valid_acc_top5
