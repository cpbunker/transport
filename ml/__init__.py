'''
Christian Bunker
M^2QM at UF
November 2021

Machine learning routines
'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn

#############################################################################
#### unsupervised learning

def Scale(X):

    from sklearn import preprocessing
    model = sklearn.preprocessing.StandardScaler();
    return model.fit_transform(X);

def PCA(X, N, inverse = False, explained = False):
    '''
    Find the first N principal components of a features matrix

    Args:
    X, ndarray, features matrix
    N, int, number of principal components
    inverse, bool, whether to inverse transform the data back to orig vector
        space after PCA decomp
    explained, bool, whether to return explained variance
    '''

    from sklearn import decomposition

    # check inputs
    assert( N < np.shape(X)[1]);

    model = sklearn.decomposition.PCA(n_components = N);
    model.fit(X);
    Xpca = model.transform(X); # reduce to N dims
    if(inverse):
        Xpca = model.inverse_transform(Xpca); # back to all dims
    elif(explained):
        Xpca = model.explained_variance_ratio; 
    return Xpca;


#############################################################################
#### regression

def Linear(X1, y1, degree = 1, intercept = True):

    from sklearn import linear_model, preprocessing

    # nothing in place
    X, y = np.copy(X1), np.copy(y1);

    # X must be a 2d arr
    if( len(np.shape(X)) == 1):
        X = X[:, np.newaxis];

    #print(np.shape(X),"\n",X[:2]);
    polynomial = sklearn.preprocessing.PolynomialFeatures(degree, include_bias = False);
    X = polynomial.fit_transform(X); # in place
    #print(np.shape(X),"\n",X[:2]);
    model = sklearn.linear_model.LinearRegression(fit_intercept = intercept);
    model.fit(X, y);
    return model.coef_, model.intercept_;


def RandomForest(X1, y1, N = 100):
    '''
    Implement random forest model, classifier for discrete = True, else regressor
    Args:
    X, ndarray, features matrix
    y, ndarray, target array
    N, int, number of classifiers
    '''

    from sklearn import ensemble

    # nothing in place
    X, y = np.copy(X1), np.copy(y1);

    model = sklearn.ensemble.RandomForestRegressor(n_estimators = N);
    model.fit(X, y);
    return model.predict(X);


#############################################################################
#### visualization
colors = plt.cm.get_cmap('tab10').colors;

def PlotPolyFit(X1, y1, degree, labs, mytitle = "Polynomial fit"):

    # nothing in place
    X, y = np.copy(X1), np.copy(y1);
    
    # check inputs
    nsamples, nfeatures = np.shape(X);
    assert(np.shape(y) == (nsamples,) );
    assert(isinstance(degree, int));

    # visualize w linear regression
    fig, axes = plt.subplots(nfeatures, sharex = True, sharey = True);
    for fi in range(nfeatures):

        # linear fit
        m, b = Linear(X[:,fi], y);
        axes[fi].scatter(X[:,fi], y, color = colors[fi]);
        fit = b*np.ones_like(X[:,fi]);
        for mi in range(len(m)): fit += m[mi]*np.power(X[:,fi], mi+1);
        fitargs = np.argsort(fit);
        axes[fi].plot(X[:,fi][fitargs], fit[fitargs], color = colors[fi] );
        axes[fi].set_ylabel("Peak");
        axes[fi].set_xlabel(labs[fi]);

    axes[0].set_title(mytitle);
    plt.show();
    return;


def PlotForestFit(X1, y1, N, labs, Nxpts, mytitle = "Random Forest fit"):

    # check inputs
    nsamples, nfeatures = np.shape(X1);
    assert(np.shape(y1) == (nsamples,) );
    assert(len(y1) % Nxpts == 0);

    # nothing in place
    X, y = np.copy(X1), np.copy(y1);

    # fit
    fit = RandomForest(X, y, N);

    # visualize 
    fig, axes = plt.subplots(nfeatures, sharex = True, sharey = True);
    for fi in range(nfeatures):

        # violin plot raw data
        # are only Nxpts distinct feature vals in pop
        # sort into batches with same xpt
        xpts = []; 
        for xpt in X[:,fi]:
            if(len(xpts) == Nxpts): break;
            if xpt not in xpts: xpts.append(xpt);
        xvals, yvals, fitvals = [], [], [];
        for _ in range(Nxpts):
            xvals.append([]);
            yvals.append([]);
            fitvals.append([]);
        for xi in range(len(X[:,fi])): # sort each sample
            whichcol = xpts.index(X[xi,fi]);
            xvals[whichcol].append(X[xi,fi]);
            yvals[whichcol].append(y[xi]);
            fitvals[whichcol].append(fit[xi]);
        xvals = np.array(xvals).T;
        yvals = np.array(yvals).T;
        axes[fi].violinplot(yvals, positions = xpts);

        # use random forest regression
        axes[fi].boxplot(fitvals, showfliers = False, positions = xpts);
        axes[fi].set_ylabel("Peak");
        axes[fi].set_xlabel(labs[fi]);

    axes[0].set_title(mytitle);
    plt.show();
    return;
    
