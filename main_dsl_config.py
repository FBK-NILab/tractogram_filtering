[DEFAULT]
########### model ###########
model = dec
batch_norm = y
bn_decay = n
bn_decay_init = 0.5
bn_decay_step = 90
bn_decay_gamma = 0.5
dropout = y
spatial_tn = 2
knngraph = 5

########## training ###########
n_epochs = 1000
optimizer = adam
accumulation_interval = n
# optimizer alternatives:
# sgd, sgd_momentum, adam
lr_type = step
learning_rate = 1e-3
min_lr = 1e-6
lr_ep_step = 90
lr_gamma = 0.7
momentum = 0.9
patience = 100
weight_decay = 5e-4
weight_init = n

########## loss ###########
loss = nll
nll_w = n
val_in_train = y
val_freq = 20

########### general ###########
n_workers = 6
model_dir = models
save_model = y
seed = 10
save_pred = n

########### logging ###########
verbose = n
print_bwgraph = y

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[HCP20]
########### data ###########
dataset = hcp20_graph
dataset_dir = /home/pa/data/ExTractor_PRIVATE/derivatives/merge_shuffle_trk
fixed_size = 7000
val_dataset_dir = /home/pa/data/ExTractor_PRIVATE/derivatives/merge_shuffle_trk
sub_list_train = data/sub_list_HCP_train.txt
sub_list_val = data/sub_list_HCP_val.txt
sub_list_test = data/sub_list_HCP_test.txt
act = y
data_dim = 3
embedding_size = 40
fold_size = 2

batch_size = 2
repeat_sampling = 3
shuffling = y
rnd_sampling = y
standardization = n
n_classes = 2
multi_category = n
ignore_class = 0
same_size = y

#experiment_name = decseq1_loss_nll-data_hcp20_resampled16_full_nogradacc_
experiment_name = pointnet_loss-nll_data-hcp20_16pts_fs7000_nobug

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************
