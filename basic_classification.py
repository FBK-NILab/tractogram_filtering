import numpy as np
from nilab.load_trk import load_streamlines
from streamline_embedding import (embed_flattened,
                                  embed_flattened_plus_flipped,
                                  embed_ordered)
from dipy.tracking.streamline import set_number_of_points
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import (KNeighborsClassifier,
                               RadiusNeighborsClassifier)
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    np.random.seed(0)
    nb_points = 16
    subset_size = 100000
    filename_template = 'data_emanuele/%s_%s_tracks.trk'
    subjects = ['100206', '100307']
    label_y = {'plausible': 1, 'unplausible': 0}

    X = []
    y = []
    groups = []
    for subject in subjects:
        for label in label_y.keys():
            filename = filename_template % (subject, label)
            streamlines, header, lengths, idxs = load_streamlines(filename,
                                                                  idxs=subset_size,
                                                                  verbose=True)

            streamlines = set_number_of_points(streamlines, nb_points)
            # tmp = embed_flattened(streamlines)
            tmp = embed_flattened_plus_flipped(streamlines)
            # tmp = embed_ordered(streamlines)
            X.append(tmp)
            y.append([label_y[label]] * len(tmp))
            groups.append([subject] * len(tmp))

    X = np.vstack(X)
    y = np.concatenate(y)
    groups = np.concatenate(groups)
    print('X.shape: %s' % (X.shape,))
    print('y.shape: %s' % (y.shape,))
    print('groups.shape: %s' % (groups.shape,))

    k = 11
    # RadiusNeighborsClassifier IS NOT WORKING BECAUSE radius IS PROBLEMATIC!
    # radius = 30.0
    # clf = RadiusNeighborsClassifier(radius=radius,
    #                                 algorithm='kd_tree', n_jobs=-1)
    n_estimators = 100
    clfs = [LogisticRegression(),
            KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree',
                                 weights='distance', n_jobs=-1),
            DecisionTreeClassifier(),
            RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)]

    cv = LeaveOneGroupOut()
    for clf in clfs:
        print(clf)
        scores = cross_validate(clf, X, y, cv=cv, groups=groups,
                                verbose=10, n_jobs=1)
        print('scores: %s' % (scores,))
