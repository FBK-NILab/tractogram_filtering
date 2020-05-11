import numpy as np
import pickle
import nibabel as nib
import argparse
import os
import csv
import glob
import sys
import multiprocessing
from functools import partial
from contextlib import contextmanager

import visdom
from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot as plt
# import plotly.plotly as py
# import plotly.graph_objs as go

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    colors = [
        "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "000000",
        "800000", "008000", "000080", "808000", "800080", "008080", "808080",
        "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0",
        "400000", "004000", "000040", "404000", "400040", "004040", "404040",
        "200000", "002000", "000020", "202000", "200020", "002020", "202020",
        "600000", "006000", "000060", "606000", "600060", "006060", "606060",
        "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0",
        "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0",
        ]

    colors = colors[:n]

    return [(int(i[:2], 16) / 255., int(i[2:4], 16) / 255., int(i[4:], 16) / 255.) for i in colors]

def plot_clustering(title, points, num_class, pred, merge=False, soft=False):
    vis = visdom.Visdom(env='clustering')

    for i in range(1, num_class+1):
        if merge:
            labels = pred
        else:
            if not soft:
                labels = ((pred == i) * i)+1
            else:
                labels = pred[:,i-1]
                # labels = np.array(range(256)).repeat(27)[:6890]
                # labels = (pred[:,i-1] * 255).astype(np.uint8)

        if not soft:
            vis.scatter(points,
                        labels,
                        opts=dict(
                            markersize=5,
                            title=title,
                            legend=range(1,num_class+1))
                        )
        else:
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(points[:,0], points[:,1], points[:,2], c=labels)
            # vis.matplot(fig)

            # data = go.Scatter3d(
            #             x=pred[:, 0],
            #             y=pred[:, 1],
            #             z=pred[:, 2],
            #             mode='markers',
            #             marker=dict(
            #                 size=3,
            #                 color=labels,  # set color to an array/list of desired values
            #                 colorscale='Viridis',   # choose a colorscale
            #                 opacity=0.8
            #             )
            #         )
            # layout = go.Layout(
            #     margin=dict(
            #         l=0,
            #         r=0,
            #         b=0,
            #         t=0
            #     )
            # )
            # fig = go.Figure(data=[data], layout=layout)
            # vis.plotlyplot(fig)
            colormap = cm.get_cmap('viridis')
            vis.scatter(points,
                        opts=dict(
                            markersize=5,
                            markercolor=(colormap(labels)[:,:3] * 255).astype(np.uint8),
                            title=title),
                        )
        if merge:
            return


def plot_tract(bundle, affine, num_class=1, pred=[], ignore=[]):
    # Visualize the results
    from dipy.viz import fvtk, actor
    from dipy.tracking.streamline import transform_streamlines

    # Create renderer
    r = fvtk.ren()

    if len(pred) > 0:
        colors = get_spaced_colors(num_class)
        for i in range(num_class):
            if i in ignore:
                continue
            idx = np.argwhere(pred == i).squeeze()
            bundle_native = transform_streamlines(
                bundle[idx], np.linalg.inv(affine))
            if len(bundle_native) == 0:
                continue
            lineactor = actor.line(
                bundle_native, colors[i], linewidth=0.2)
            fvtk.add(r, lineactor)
    elif num_class == 1:
        bundle_native = transform_streamlines(bundle, np.linalg.inv(affine))
        lineactor = actor.line(bundle_native, linewidth=0.2)
        fvtk.add(r, lineactor)

    # Show original fibers
    fvtk.camera(r, pos=(-264, 285, 155), focal=(0, -14, 9),
        viewup=(0, 0, 1), verbose=False)

    fvtk.show(r)


if __name__ == '__main__':


    #### ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument('-viz_pred_bundle', nargs='*', default='',
                        help='path to the prediction file and path to the trk file to be filtered')
    parser.add_argument('-compute_pred_trk', nargs='*', default='',
                        help='path to the prediction file and path to the trk file to be filtered')
    parser.add_argument('-viz_sl_clusters', nargs='*', default='',
                        help='path to the prediction file and path to the trk file to visualize')
    parser.add_argument('-viz_clusters', nargs='*', default='',
                        help='path to the prediction file and path to the shape(points) to visualize')
    parser.add_argument('-viz_multiclusters', nargs='*', default='',
                        help='3 arguments required: path to the prediction dir, path to the shape dir, number of shapes to show')
    parser.add_argument('--gt', nargs='?', const=1,
                        help='path to the gt pickle file')
    parser.add_argument('--merge', nargs='?', const=1, default=False,
                        help='plot an unique shape containing all the labels')
    parser.add_argument('--soft', nargs='?', const=1, default=False,
                        help='plot softmax output with percentage instead of hard labeling')
    args = parser.parse_args()


    #### CONFIG PARSING

    # Reading configuration file with specific setting for a run.
    # Mandatory variables for this script:

    if args.viz_pred_bundle:
        pred_file = args.viz_pred_bundle[0]
        trk_file = args.viz_pred_bundle[1]

        import ipdb; ipdb.set_trace()
        with open(pred_file, 'rb') as f:
            pred = pickle.load(f)
        trk = nib.streamlines.load(trk_file)

        if args.gt:
            pred = np.argmax(pred,axis=1)

            with open(args.gt, 'rb') as f:
                gt = np.array(pickle.load(f))
            tp = (pred * gt) == 1
            fp = 2 * ((pred - gt) == 1)
            fn = 3 * ((pred - gt) == -1)

            pred = tp + fp + fn
            plot_tract(trk.streamlines, trk.affine,
                       num_class=4, pred=pred, ignore=[0])
        else:
            plot_tract(trk.streamlines[pred], trk.affine)

    if args.compute_pred_trk:
        pred_file = args.compute_pred_trk[0]
        trk_file = args.compute_pred_trk[1]

        with open(pred_file, 'rb') as f:
            pred = pickle.load(f)
        trk = nib.streamlines.load(trk_file)

        hdr = trk.header.copy()
        hdr['nb_streamlines'] = len(pred)
        outfile = os.path.join(pred_file.rsplit('/',1)[0],
                                trk_file.split('/')[-1])
        nib.streamlines.save(nib.streamlines.Tractogram(trk.streamlines[pred],
                                                    affine_to_rasmm=np.eye(4)),
                                                    outfile,
                                                    header=hdr)

    # if args.viz_shape_pred:
    #     pred_file = args.viz_cluster[0]
    #     data_file = args.viz_cluster[1]

    #     with open(pred_file, 'rb') as f:
    #         pred_idx = pickle.load(f)
    #     with open(data_file, 'rb') as f:
    #         reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    #         points = [row[:3] for row in reader]
    #     points = np.array(points)

    #     pred = np.zeros_like(points)
    #     for i in range(len(pred_idx)):
    #         pred[pred_idx[i]] = i+1

    #     if args.merge:
    #         merge = True

    #     plot_clustering(points, len(pred_idx), pred, merge=merge)

    if args.viz_clusters:
        pred_file = args.viz_clusters[0]
        data_file = args.viz_clusters[1]
        with open(pred_file, 'rb') as f:
            sm_out = pickle.load(f)
        with open(data_file, 'rb') as f:
            if '.pts' in data_file:
                reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
            else:
                reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            points = [row[:3] for row in reader]
        points = np.array(points)
        import ipdb; ipdb.set_trace()
        if len(sm_out) > 100:
            sm_out = [sm_out]
        for l in range(len(sm_out)):
            sm_out[l] = sm_out[l].squeeze()
            if args.soft:
                pred = sm_out[l]
                print(pred.max())
            else:
                pred = np.argmax(sm_out[l],axis=1) + 1
            print(pred.shape)
            print(points.shape)

            title = 'layer%d_' % l + '/'.join(pred_file.rsplit('/',3)[-3:])
            plot_clustering(title,
                            points,
                            sm_out[l].shape[1],
                            pred,
                            merge=args.merge,
                            soft=args.soft)

    if args.viz_multiclusters:
        if len(args.viz_multiclusters) < 2:
            sys.exit('ERORR: missing arguments')
        else:
            pred_dir = args.viz_multiclusters[0]
            data_dir = args.viz_multiclusters[1]

        if len(args.viz_multiclusters) == 2:
            n_files = 1
        else:
            n_files = args.viz_multiclusters[2]
        pred_files = glob.glob(pred_dir + '/*sm_1*')
        pred_files.sort()
        for i in range(int(n_files)):
            pred_file = pred_files[i]
            print(pred_file.rsplit('/', 1)[1].rsplit('_',2)[0])
            sample_name = pred_file.rsplit('/', 1)[1].rsplit('_',2)[0]
            data_file = glob.glob(os.path.join(
                                            data_dir,
                                            sample_name + '*'))[0]
            with open(pred_file, 'rb') as f:
                sm_out = pickle.load(f)
            with open(data_file, 'rb') as f:
                if '.pts' in data_file:
                    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                else:
                    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                points = [row[:3] for row in reader]
            points = np.array(points)
            import ipdb; ipdb.set_trace()
            if len(sm_out) > 100:
                sm_out = [sm_out]
            for l in range(len(sm_out)):
                sm_out[l] = sm_out[l].squeeze()
                if args.soft:
                    pred = sm_out[l]
                    print(pred.max())
                else:
                    pred = np.argmax(sm_out[l],axis=1) + 1
                print(pred.shape)
                print(points.shape)

                title = 'layer%d_' % l + '/'.join(pred_file.rsplit('/',3)[-3:])
                plot_clustering(title,
                                points,
                                sm_out[l].shape[1],
                                pred,
                                merge=args.merge,
                                soft=args.soft)

    if args.viz_sl_clusters:
        pred_file = args.viz_sl_clusters[0]
        trk_file = args.viz_sl_clusters[1]

        with open(pred_file, 'rb') as f:
            sm_out = np.array(pickle.load(f)).squeeze()

        trk = nib.streamlines.load(trk_file)

        pred = np.argmax(sm_out,axis=1)

        if not args.merge:
            clusters = []
            for i in range(sm_out.shape[1]):
                idx = (np.argwhere(pred == i).squeeze())
                if len(idx) > 0:
                    clusters.append(trk.streamlines[idx])
            with poolcontext(processes=sm_out.shape[1]) as pool:
                pool.map(partial(plot_tract, affine=trk.affine),
                                 clusters)
        else:
            plot_tract(trk.streamlines, trk.affine,
                       num_class=sm_out.shape[1], pred=pred)
