# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import time
import random
import math
from collections import defaultdict, namedtuple


import torch
from torch.utils.data import Dataset
from torch.autograd.profiler import EventList, FunctionEventAvg, format_time, format_time_share, format_memory
from torch.autograd import DeviceType
import torch.nn as nn
import csv

from PIL import Image, ImageFilter
from typing import Dict, List, Tuple, Optional


class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def init_weights(m):
    '''Initialize weights with zeros
    '''
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


class CustomDataset(Dataset):
    """ Creates a custom pytorch dataset.

        - Creates two views of the same input used for unsupervised visual
        representational learning. (SimCLR, Moco, MocoV2)

    Args:
        data (array): Array / List of datasamples

        labels (array): Array / List of labels corresponding to the datasamples

        transforms (Dictionary, optional): The torchvision transformations
            to make to the datasamples. (Default: None)

        target_transform (Dictionary, optional): The torchvision transformations
            to make to the labels. (Default: None)

        two_crop (bool, optional): Whether to perform and return two views
            of the data input. (Default: False)

    Returns:
        img (Tensor): Datasamples to feed to the model.

        labels (Tensor): Corresponding lables to the datasamples.
    """

    def __init__(self, data, labels, transform=None, target_transform=None, two_crop=False):

        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])

        if isinstance(data, torch.Tensor):
            data = data.numpy()  # to work with `ToPILImage'

        self.data = data[idx]

        # when STL10 'unlabelled'
        if not labels is None:
            self.labels = labels[idx]
        else:
            self.labels = labels

        self.transform = transform
        self.target_transform = target_transform
        self.two_crop = two_crop

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        # If the input data is in form from torchvision.datasets.ImageFolder
        if isinstance(self.data[index][0], np.str_):
            # Load image from path
            image = Image.open(self.data[index][0]).convert('RGB')

        else:
            # Get image / numpy pixel values
            image = self.data[index]

        if self.transform is not None:

            # Data augmentation and normalisation
            img = self.transform(image)

        if self.target_transform is not None:

            # Transforms the target, i.e. object detection, segmentation
            target = self.target_transform(target)

        if self.two_crop:

            # Augments the images again to create a second view of the data
            img2 = self.transform(image)

            # Combine the views to pass to the model
            img = torch.cat([img, img2], dim=0)

        # when STL10 'unlabelled'
        if self.labels is None:
            return img, torch.Tensor([0])
        else:
            return img, self.labels[index].long()


def random_split_image_folder(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.

        Specifically for the image folder class
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    # torch.from_numpy(np.stack(labels)) this takes the list of class ids and turns them to tensor.long

    return {'train': train_x, 'valid': valid_x}, \
        {'train': torch.from_numpy(np.stack(train_y)), 'valid': torch.from_numpy(np.stack(valid_y))}


def random_split(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'train': torch.stack(train_x), 'valid': torch.stack(valid_x)}, \
            {'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'train': torch.from_numpy(np.stack(train_x)),
            'valid': torch.from_numpy(np.stack(valid_x))}, \
        {'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}


def sample_weights(labels):
    """ Calculates per sample weights. """
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]


def experiment_config(parser, args):
    """ Handles experiment configuration and creates new dirs for model.
    """
    # check number of models already saved in 'experiments' dir, add 1 to get new model number
    run_dir = os.path.join(os.path.split(os.getcwd())[0], 'experiments')

    os.makedirs(run_dir, exist_ok=True)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # create all save dirs
    args.model_dir = os.path.join(run_dir, run_name)

    os.makedirs(args.model_dir, exist_ok=True)

    args.summaries_dir = os.path.join(args.model_dir, 'summaries')
    args.checkpoint_dir = os.path.join(args.model_dir, 'checkpoint.pt')

    if not args.finetune:
        args.load_checkpoint_dir = args.checkpoint_dir

    os.makedirs(args.summaries_dir, exist_ok=True)

    # save hyperparameters in .txt file
    with open(os.path.join(args.model_dir, 'hyperparams.txt'), 'w') as logs:
        for key, value in vars(args).items():
            logs.write('--{0}={1} \n'.format(str(key), str(value)))

    # save config file used in .txt file
    with open(os.path.join(args.model_dir, 'config.txt'), 'w') as logs:
        # Remove the string from the blur_sigma value list
        config = parser.format_values().replace("'", "")
        # Remove the first line, path to original config file
        config = config[config.find('\n')+1:]
        logs.write('{}'.format(config))

    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.model_dir, 'trainlogs.txt')),
                                  logging.StreamHandler()])
    return args


def print_network(model, args):
    """ Utility for printing out a model's architecture.
    """
    logging.info('-'*70)  # print some info on architecture
    logging.info('{:>25} {:>27} {:>15}'.format('Layer.Parameter', 'Shape', 'Param#'))
    logging.info('-'*70)

    for param in model.state_dict():
        p_name = param.split('.')[-2]+'.'+param.split('.')[-1]
        # don't print batch norm layers for prettyness
        if p_name[:2] != 'BN' and p_name[:2] != 'bn':
            logging.info(
                '{:>25} {:>27} {:>15}'.format(
                    p_name,
                    str(list(model.state_dict()[param].squeeze().size())),
                    '{0:,}'.format(np.product(list(model.state_dict()[param].size())))
                )
            )
    logging.info('-'*70)

    logging.info('\nTotal params: {:,}\n\nSummaries dir: {}\n'.format(
        sum(p.numel() for p in model.parameters()),
        args.summaries_dir))

    for key, value in vars(args).items():
        if str(key) != 'print_progress':
            logging.info('--{0}: {1}'.format(str(key), str(value)))

#save_events_table(EventList_obj, sort_by=None, row_limit=100, max_src_column_width=75, header=None, top_level_events_only=False)
#might want to investigate top_level_events_only=True?? 
#key_averages returns an EventList
#the table function calls buil_table in autograd/profiler.py


def save_events_table(
        events,
        path,
        times_path=None,
        sort_by=None,
        header=None,
        row_limit=100,
        max_src_column_width=75,
        with_flops=False,
        profile_memory=False,
        top_level_events_only=False,
        only_save_root_call=False):
    """Prints a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if len(events) == 0:
        return ""

    has_cuda_time = any([event.self_cuda_time_total > 0 for event in events])
    has_cuda_mem = any([event.self_cuda_memory_usage > 0 for event in events])
    has_input_shapes = any(
        [(event.input_shapes is not None and len(event.input_shapes) > 0) for event in events])

    if sort_by is not None:
        events = EventList(sorted(
            events, key=lambda evt: getattr(evt, sort_by), reverse=True
        ), use_cuda=has_cuda_time, profile_memory=profile_memory, with_flops=with_flops)

    stacks = []
    for evt in events:
        if evt.stack is not None and len(evt.stack) > 0:
            stacks.append(evt.stack)
    has_stack = len(stacks) > 0
    print(f"Has stack: {has_stack}")

    #time columns in microseconds, memory columns in bytes

    headers = [
        'Name',
        'Self CPU %',
        'Self CPU',
        'CPU total %',
        'CPU total',
        'CPU time avg',
    ]
    if has_cuda_time:
        headers.extend([
            'Self CUDA',
            'Self CUDA %',
            'CUDA total',
            'CUDA time avg',
        ])
    if profile_memory:
        headers.extend([
            'CPU Mem',
            'Self CPU Mem',
        ])
        if has_cuda_mem:
            headers.extend([
                'CUDA Mem',
                'Self CUDA Mem',
            ])
    headers.append(
        '# of Calls'
    )
    # Only append Node ID if any event has a valid (>= 0) Node ID
    append_node_id = any([evt.node_id != -1 for evt in events])
    if append_node_id:
        headers.append('Node ID')

    # Have to use a list because nonlocal is Py3 only...
    MAX_STACK_ENTRY = 5

    def auto_scale_flops(flops):
        flop_headers = [
            'FLOPS',
            'KFLOPS',
            'MFLOPS',
            'GFLOPS',
            'TFLOPS',
            'PFLOPS',
        ]
        assert flops > 0
        log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
        assert log_flops >= 0 and log_flops < len(flop_headers)
        return (pow(10, (math.floor(log_flops) * -3.0)), flop_headers[int(log_flops)])


    with open(path, 'w') as profiler_csv:
        writer = csv.writer(profiler_csv)

        if with_flops:
            # Auto-scaling of flops header
            US_IN_SECOND = 1000.0 * 1000.0  # cpu_time_total is in us
            raw_flops = []
            for evt in events:
                if evt.flops > 0:
                    if evt.cuda_time_total != 0:
                        evt.flops = float(evt.flops) / evt.cuda_time_total * US_IN_SECOND
                    else:
                        evt.flops = float(evt.flops) / evt.cpu_time_total * US_IN_SECOND
                    raw_flops.append(evt.flops)
            if len(raw_flops) != 0:
                (flops_scale, flops_header) = auto_scale_flops(min(raw_flops))
                header = headers + flops_header
            else:
                with_flops = False  # can't find any valid flops
        if has_stack:
            headers = headers + ['Source Location']
        writer.writerow(headers)


        # Have to use a list because nonlocal is Py3 only...
        result = []

        sum_self_cpu_time_total = sum([event.self_cpu_time_total for event in events])
        sum_self_cuda_time_total = 0
        for evt in events:
            if evt.device_type == DeviceType.CPU:
                # in legacy profiler, kernel info is stored in cpu events
                if evt.is_legacy:
                    sum_self_cuda_time_total += evt.self_cuda_time_total
            elif evt.device_type == DeviceType.CUDA:
                # in kineto profiler, there're events with the correct device type (e.g. CUDA)
                sum_self_cuda_time_total += evt.self_cuda_time_total

        # Actual printing
        event_limit = 0
        for evt in events:
            if event_limit == row_limit:
                break
            if top_level_events_only and evt.cpu_parent is not None:
                continue
            else:
                event_limit += 1
            name = evt.key

            '''
            row_values = [
                name,
                # Self CPU total %, 0 for async events.
                format_time_share(evt.self_cpu_time_total,
                                sum_self_cpu_time_total),
                evt.self_cpu_time_total,  # Self CPU total
                # CPU total %, 0 for async events.
                format_time_share(evt.cpu_time_total, sum_self_cpu_time_total) if not evt.is_async else 0,
                evt.cpu_time_total,  # CPU total
                evt.cpu_time,  # CPU time avg
            ]'''
            row_values = [
                name,
                # Self CPU total %, 0 for async events.
                (evt.self_cpu_time_total/sum_self_cpu_time_total),
                evt.self_cpu_time_total,  # Self CPU total
                # CPU total %, 0 for async events.
                (evt.cpu_time_total / sum_self_cpu_time_total) if not evt.is_async else 0,
                evt.cpu_time_total,  # CPU total
                evt.cpu_time,  # CPU time avg
            ]
            if has_cuda_time:
                row_values.extend([
                    evt.self_cuda_time_total,
                    # CUDA time total %
                    (evt.self_cuda_time_total/sum_self_cuda_time_total),
                    evt.cuda_time_total,
                    evt.cuda_time,  # Cuda time avg
                ])
            if profile_memory:
                row_values.extend([
                    # CPU Mem Total
                    evt.cpu_memory_usage,
                    # Self CPU Mem Total
                    evt.self_cpu_memory_usage,
                ])
                if has_cuda_mem:
                    row_values.extend([
                        # CUDA Mem Total
                        evt.cuda_memory_usage,
                        # Self CUDA Mem Total
                        evt.self_cuda_memory_usage,
                    ])
            row_values.append(
                evt.count,  # Number of calls
            )

            if append_node_id:
                row_values.append(evt.node_id)
            if has_input_shapes:
                row_values.append(str(evt.input_shapes))
            if with_flops:
                if evt.flops <= 0.0:
                    row_values.append("--")
                else:
                    row_values.append('{0:8.3f}'.format(evt.flops * flops_scale))
            if has_stack:
                if (len(evt.stack) > 0) and only_save_root_call:
                    src_field = evt.stack[0]
                elif not only_save_root_call:
                    src_field = ",".join(evt.stack)
                row_values.append(src_field)

            writer.writerow(row_values)

    if times_path is not None:
        with open(times_path, 'w') as profiler_log:
            profiler_log.write("Self CPU time total: {}".format(format_time(sum_self_cpu_time_total)))
            profiler_log.write("Self CUDA time total: {}".format(format_time(sum_self_cuda_time_total)))


def key_averages_with_stack(evtlist, group_by_input_shapes=False, group_by_stack_n=0):
    """Averages all function events over their keys.
        Args:
            group_by_input_shapes: group entries by
                (event name, input shapes) rather than just event name.
                This is useful to see which input shapes contribute to the runtime
                the most and may help with size-specific optimizations or
                choosing the best candidates for quantization (aka fitting a roof line)
            group_by_stack_n: group by top n stack trace entries
        Returns:
            An EventList containing FunctionEventAvg objects.
        """

    assert evtlist._tree_built
    stats: Dict[Tuple[str, ...], FunctionEventAvg] = defaultdict(FunctionEventAvg)

    def get_key(event, group_by_input_shapes, group_by_stack_n) -> Tuple[str, ...]:
        key = [str(event.key), str(event.node_id), str(event.device_type), str(event.is_legacy)]
        if group_by_input_shapes:
            key.append(str(event.input_shapes))
        if group_by_stack_n > 0:
            key += event.stack[:group_by_stack_n]
        return tuple(key)
    for evt in evtlist:
        stats[get_key(evt, group_by_input_shapes, group_by_stack_n)].add(evt)

    avg_list = EventList(
        stats.values(),
        use_cuda=evtlist._use_cuda,
        profile_memory=evtlist._profile_memory,
        with_flops=evtlist._with_flops)
    for evt in avg_list:
        if group_by_stack_n > 0:
            evt.stack = evt.stack[:group_by_stack_n]
        if not group_by_input_shapes:
            evt.input_shapes = ""
    return avg_list