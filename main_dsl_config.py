[DEFAULT]
########### local embedding ###########
embedding_type = dr
# dr: dissimilarity representation
dr_distance = mam
prototypes_file = data/prototypes.trk
# prototypes.trk saved in voxel coordinates

########### model ###########
model = dec
batch_norm = y
bn_decay = y
bn_decay_init = 0.5
bn_decay_step = 90
bn_decay_gamma = 0.5
dropout = y
spatial_tn = 2
gf_operation = max
num_gf = 10
dyn_k = n
soft_gf = 1
centroids_gf = n
simple = n
simple_size = 128
barycenter = n
feat_scaling = n
direct_clustering = y
full_concat = n
multi_layer = n
n_layers = 1
residual = n
direct_pooling = n
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
clip_gradients = n
weight_init = n

########## loss ###########
loss = nll
nll_w = n
switch_loss_epoch = 2500
LLm_op = mean
# loss legend:
#   - nll: negative log likelihood
#   - LLh: Lovàsz loss hinge(binary)
#   - LLm: Lovàsz loss multiclass
multi_loss = n
cluster_loss = center
alfa_loss = 4e-4
split_backprop = n
val_in_train = y
val_freq = 20

########### general ###########
load_all_once = y
n_workers = 8
model_dir = models
save_model = y
seed = 10
save_pred = n
save_softmax_out = n
save_gf = n

########### logging ###########
verbose = n
viz_emb_input = n
viz_logits_distribution = n
viz_learned_features = n
print_bwgraph = n
viz_clusters = n
save_embedding = n
#ed73e946a138f3cfbc0909d98a1ff2b4
# graph_type = dense

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[IFOF]
########### data ###########
dataset = left_ifof_ss_sl
dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/bundle_filtering_STEM
fixed_size = 500
val_dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/bundle_filtering_STEM
data_dim = 3
sub_list_train = data/sub_list_STEM-FNALMBE_train.txt
sub_list_val = data/sub_list_STEM-FNALMBE_val.txt

batch_size = 1
shuffling = y
rnd_sampling = y
standardization= n
embedding_size = 40
n_classes = 2
multi_category = n

experiment_name = pnCLS-loss_nll-data_ifof

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[IFOF_GRAPH]
########### data ###########
dataset = left_ifof_ss_sl_graph
dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/streamlines_resampling_BSPLINE
#dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/bundle_filtering_STEM
fixed_size = 100
#val_dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/bundle_filtering_STEM
val_dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/streamlines_resampling_BSPLINE
data_dim = 3
sub_list_train = data/sub_list_STEM-FNALMBE_train.txt
sub_list_val = data/sub_list_STEM-FNALMBE_val.txt

batch_size = 10
same_size = y
shuffling = y
rnd_sampling = y
standardization= n
embedding_size = 40
n_classes = 2
multi_category = n

experiment_name = 2unit-gcn_gcn_gradientCheck-loss_nll-data_bspline-ifof

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[IFOF_EMB]
########### data ###########
dataset = left_ifof_emb
emb_dataset_dir = /home/pietro/github/deep-streamlines-learning/runs/gcn_nodropout-loss_nll-data_ifof_0/embedding
gt_dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/dissimilarity_representation_STEM-FNALMBE
fixed_size = 6000
data_dim = 40
sub_list_train = data/sub_list_STEM-FNALMBE_train.txt
sub_list_val = data/sub_list_STEM-FNALMBE_val.txt

batch_size = 16
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 2
multi_category = n
precompute_graph = n

experiment_name = gcnseg-loss_nll-data_gcnemb-ifof

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[IFOF_DR]
########### data ###########
dataset = left_ifof_ss_dr
dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/dissimilarity_representation_SS2kDR
fixed_size = 10000
val_dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/dissimilarity_representation_SS2kDR
data_dim = 40
sub_list_train = data/sub_list_2k_train.txt
sub_list_val = data/sub_list_2k_test.txt

batch_size = 5
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 2
multi_category = n

experiment_name = pointnet_local-loss_nll-data_ifof

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[IFOFSTEM]
########### data ###########
dataset = left_ifof_ss_dr
dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/dissimilarity_representation_STEM-FNALMBE
fixed_size = 7000
val_dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/dissimilarity_representation_STEM-FNALMBE
data_dim = 40
sub_list_train = data/sub_list_STEM-FNALMBE_train.txt
sub_list_val = data/sub_list_STEM-FNALMBE_val.txt

batch_size = 16
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 2
multi_category = n

experiment_name = gcnseg-loss_nll-data_ifof

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[IFOFSTEM_APSS]
########### data ###########
dataset = left_ifof_ss_dr
dataset_dir =
fixed_size = 10000
val_dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/dissimilarity_representation_CFACUTTRACED-W5WL470
data_dim = 40

standardization= n
n_classes = 2
multi_category = n

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[HCP20]
########### data ###########
dataset = hcp20_graph
dataset_dir = /home/pa/data/ExTractor_PRIVATE/derivatives/merge_shuffle_trk
fixed_size = 8000
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

experiment_name = dec_own-loss_nll-data_hcp20_full_nogradacc

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[TRACTSEG]
########### data ###########
dataset = tractseg_500k
dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/downsampled_tractograms
fixed_size = 3740
val_dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/downsampled_tractograms
sub_list_train = data/sub_list_tractseg_train_reduced.txt
sub_list_val = data/sub_list_tractseg_val.txt
act = y
data_dim = 3
embedding_size = 40
fold_size = 2

batch_size = 5
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 68
multi_category = n
ignore_class = 0

experiment_name = gcn-loss_nll-data_tractseg

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[TRACTSEG_DR]
########### data ###########
dataset = tractseg_500k_dr
dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/dissimilarity_representation_TS500KDR
fixed_size = 10000
val_dataset_dir = /home/pietro/datasets/dsl_dataset/derivatives/dissimilarity_representation_TS500KDR
sub_list_train = data/sub_list_tractseg_train_reduced.txt
sub_list_val = data/sub_list_tractseg_val.txt
data_dim = 40
act = y

batch_size = 1
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 7
multi_category = n
ignore_class = 0

experiment_name = gcnseg_accGrad20-loss_nll-data_tractseg

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************
[HUMANASYM]
########### data ###########
dataset = shapes
dataset_dir = /home/pietro/datasets/dsl_dataset/benchmarks/FAUST_asym
fixed_size = 1000
data_dim = 6

batch_size = 8
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 10
multi_category = n

experiment_name = pointnetmgf_max_10gf-loss_LLm-loss_LLm-data_faustAsym

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[HUMANSYM]
########### data ###########
dataset = shapes
dataset_dir = /home/pietro/datasets/dsl_dataset/benchmarks/FAUST_sym
fixed_size = 1000
data_dim = 6

batch_size = 8
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 6
multi_category = n

experiment_name = pointnetmgf_max_10gf-loss_LLm-data_faustSym

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[AIRPLANE]
########### data ###########
dataset = shapes
dataset_dir = /home/pietro/datasets/dsl_dataset/benchmarks/psbAirplanev2
fixed_size = 5000
data_dim = 6

batch_size = 3
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 5
multi_category = n

experiment_name = pointnetmgfml_mean_10gf_soft_fsINrebn_resizeDC_10l_simple-loss_LLm_mean-data_psbAirplanev2

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[TEDDY]
########### data ###########
dataset = shapes
dataset_dir = /home/pietro/datasets/dsl_dataset/benchmarks/psbTeddy
fixed_size = 5000
data_dim = 6

batch_size = 3
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 5
multi_category = n

experiment_name = pointnetmgfml_max_10gf_2l_resizeDCnew_simple-loss_LLm_mean-data_teddy

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[ARMADILLO]
########### data ###########
dataset = shapes
dataset_dir = /home/pietro/datasets/dsl_dataset/benchmarks/psbArmadillo_5K
fixed_size = 5000
data_dim = 6

batch_size = 3
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 11
multi_category = n

experiment_name = pointnetmgfml_mean_10gf_soft_fs0learnGF_resizeDCnew_dropoutLast_11l_simple-loss_LLm_mean-data_armadillo

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[4LEGS]
########### data ###########
dataset = shapes
dataset_dir = /home/pietro/datasets/dsl_dataset/benchmarks/psbFourLeg
fixed_size = 5000
data_dim = 6

batch_size = 3
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 6
multi_category = n

experiment_name = pointnet_local-loss_LLm_mean-data_4legs

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[MULTI]
########### data ###########
dataset = shapes
dataset_dir = /home/pietro/datasets/dsl_dataset/benchmarks/psbMulti
fixed_size = 5000
data_dim = 6

batch_size = 8
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 27
multi_category = 4

experiment_name = tmp
#pointnetmgfml_mean_10gf_soft_fs0learnGF _resizeDCnew_10l_dropoutLast_simple-loss_LLm_mean-data_multi

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[SN_AIRPLANE]
########### data ###########
dataset = shapenet
dataset_dir = /home/pietro/datasets/dsl_dataset/benchmarks/shapenetcore_partanno_segmentation_benchmark_v0/02691156
fixed_size = 1000
data_dim = 3

batch_size = 32
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 4
multi_category = n

experiment_name = tmp
#pointnetmgf_mean_10gfCFid1_switch-loss_nll_center-data_snAirplane
#pointnetmgf_mean_10gfCF_softTemp_simple256all_newLR-loss_nll-data_snAirplane

[SN_FULL]
########### data ###########
dataset = shapenet
dataset_dir = /home/pietro/datasets/dsl_dataset/benchmarks/shapenetcore_partanno_segmentation_benchmark_v0
fixed_size = 1000
data_dim = 3

batch_size = 32
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 50
multi_category = 16
num_parts = 4 2 2 4 4 3 3 2 4 2 6 2 3 3 3 3

experiment_name = pointnetmgf_max_10gf_soft-loss_LLm-data_snFull

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[MN40]
########### data ###########
dataset = modelnet
dataset_dir = /home/pietro/datasets/modelnet40_ply_hdf5_2048
fixed_size = 2500
data_dim = 3

batch_size = 32
shuffling = y
rnd_sampling = n
standardization= n
n_classes = 40
multi_category = n

experiment_name = tmp_finalConv1d-data_mn40
#pointnetmgf_max_10gf_soft-loss_LLm-data_snFull

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[SCANOBJ]
########### data ###########
dataset = scanobj
dataset_dir = /home/pietro/datasets/dsl_dataset/benchmarks/ScanObjectNN/h5_files/main_split
fixed_size = 1024
data_dim = 3
scanobj_variant = hard
scanobj_bg = y

batch_size = 32
shuffling = y
rnd_sampling = y
standardization= n
n_classes = 15
multi_category = n

#experiment_name = tmp_finalConv1d-data_scanobj
experiment_name = pointnetcls_2ST-loss_nll_ST-data_scanobj_hard_bg
