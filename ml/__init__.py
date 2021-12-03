'''
Christian Bunker
M^2QM at UF
November 2021

Machine learning routines
'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn

#############################################################################
#### unsupervised learning

def Scale(X):

    from sklearn import preprocessing
    model = sklearn.preprocessing.StandardScaler();
    return model.fit_transform(X);

def PCA(X, N, inverse = False, explained = False, components = False):
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
    assert( N <= np.shape(X)[1]);

    model = sklearn.decomposition.PCA(n_components = N);
    model.fit(X);
    Xpca = model.transform(X); # reduce to N dims
    if(explained):
        return model.explained_variance_ratio_;
    if(components):
        args = [];
        for Ni in range(N): # iter over pca components
            args.append(np.argmax(abs(model.components_[Ni])));
            print("Component "+str(Ni)+str(model.components_[Ni]));
        return args;
    if(inverse):
        Xpca = model.inverse_transform(Xpca); # back to all dims
    return Xpca;


def LLE(X, N, nn):

    from sklearn import manifold

    model = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=nn, n_components=N, method='modified',eigen_solver='dense');
    return model.fit_transform(X);


#############################################################################
#### classification


def GMM(X1, y1, N, score = False, density = False):

    from sklearn import mixture, metrics, model_selection

    # nothing in place
    X, y = np.copy(X1), np.copy(y1);
    nfeatures, nsamples = np.shape(X);

    # GMM model
    model = sklearn.mixture.GaussianMixture(n_components = N, covariance_type='full', random_state=0);

    # score
    model.fit(X);
    scores = model.bic(X);

    if score:
        return scores;

    # find regions of high density
    nsamps = 1000
    if density:
        new, _ = model.sample(n_samples = nsamps);
        dense_scores = [];
        for si in range(len(new)): # score density by minimizing nn distances
            distances = [];
            for sj in range(len(new)):
                distances.append(np.dot(new[si] - new[sj], new[si] - new[sj]));
            distances = np.sort(distances)[:nfeatures]; # nn only
            dense_scores.append(-sum(distances));

        maxpt = new[np.argmax(dense_scores)];
        return maxpt;
    

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
colors = seaborn.color_palette("dark");

def PlotPolyFit(X1, y1, degree, labs, mytitle = "Polynomial fit"):

    plt.style.use("seaborn-dark-palette");

    # nothing in place
    X, y = np.copy(X1), np.copy(y1);
    
    # check inputs
    nsamples, nfeatures = np.shape(X);
    assert(np.shape(y) == (nsamples,) );
    assert(len(labs) == nfeatures);
    assert(isinstance(degree, int));

    # visualize w linear regression
    fig, axes = plt.subplots(nfeatures, sharex = True, sharey = True);
    if(nfeatures == 1): axes = [axes];
    for fi in range(nfeatures):

        # linear fit
        m, b = Linear(X[:,fi], y);
        axes[fi].scatter(X[:,fi], y, marker = 's', color = colors[fi]);
        fit = b*np.ones_like(X[:,fi]);
        for mi in range(len(m)): fit += m[mi]*np.power(X[:,fi], mi+1);
        fitargs = np.argsort(fit);
        #axes[fi].plot(X[:,fi][fitargs], fit[fitargs], color = colors[fi] );
        axes[fi].set_xlabel(labs[fi]);

    axes[0].set_title(mytitle);
    axes[nfeatures // 2].set_ylabel("Entanglement prob.");
    plt.show();
    return;


def PlotTarget2d(X1, y1, labs):
    
    # nothing in place
    X, y = np.copy(X1), np.copy(y1);

    # unpack
    nsamples, nfeatures = np.shape(X);
    print("nsamples, nfeatures", (nsamples, nfeatures));

    # bin samples with identical features together
    Xpts = [];
    for si in range(nsamples):
        included = False;
        for el in Xpts:
            if( not np.any(X[si] - el)): # true if they are equal
                included = True;
        if( not included):
            Xpts.append(X[si]);
    Xpts = np.array(Xpts);

    # sort into bins
    xvals = [];
    yvals = [];
    for _ in range(len(Xpts)):
        xvals.append([]);
        yvals.append([]);
    for si in range(nsamples):
        for eli in range(len(Xpts)):
            if( not np.any(X[si] - Xpts[eli])):
                whichcol = eli;
        xvals[whichcol].append(X[si]);
        yvals[whichcol].append(y[si]);
    xvals = np.array(xvals);
    yvals = np.array(yvals);

    # make each bin single valued
    xvals = xvals[:,0,:];
    yvals = np.amax(yvals, axis = 1);

    # plot
    fig, ax = plt.subplots()
    scatterplot = ax.scatter(xvals[:,0], xvals[:,1], marker = 's', s = 1000, c = yvals, cmap = 'seismic', vmin = 0.0, vmax = 0.2);
    ax.set_xlabel(labs[0]);
    ax.set_ylabel(labs[1]);
    ax.set_title("Entanglement vs. principal parameters");
    colorbar = fig.colorbar(scatterplot);
    colorbar.set_label("Entanglement prob.");
    plt.show();

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    print(">>>",fig.dpi*bbox.width, fig.dpi*bbox.height);
    


def PlotForestFit(X1, y1, N, labs, Nxpts, mytitle = "Random Forest fit", box = False):

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

        # use random forest regression to model dist
        if box:
            axes[fi].boxplot(fitvals, showfliers = False, positions = xpts);

        # format
        axes[fi].set_ylabel("Entanglement prob.");
        axes[fi].set_xlabel(labs[fi]);

    axes[0].set_title(mytitle);
    plt.show();
    return;


def PlotBinsViolin(X1, y1, labs):

    # nothing in place
    X, y = np.copy(X1), np.copy(y1);

    # unpack
    nsamples, nfeatures = np.shape(X);
    print("nsamples, nfeatures", (nsamples, nfeatures));

    # bin samples with identical features together
    Xpts = [];
    for si in range(nsamples):
        included = False;
        for el in Xpts:
            if( not np.any(X[si] - el)): # true if they are equal
                included = True;
        if( not included):
            Xpts.append(X[si]);

    # sort into bins
    yvals = [];
    for _ in range(len(Xpts)):
        yvals.append([]);
    for si in range(nsamples):
        for eli in range(len(Xpts)):
            if( not np.any(X[si] - Xpts[eli])):
                whichcol = eli;
        yvals[whichcol].append(y[si]);
    yvals = np.array(yvals);
    
    # show
    indices = [np.argmax(np.mean(yvals, axis = 1)), np.argmin(np.mean(yvals, axis = 1))];
    fig, axes = plt.subplots(ncols = len(indices), sharey = True);
    for i in range(len(indices)):
        axes[i].violinplot(yvals[indices[i]], showextrema = False);
        axes[i].scatter([1], [np.median(yvals[indices[i]])], color = "white", s = 50, marker = 's');
        axes[i].scatter([1,1], [np.max(yvals[indices[i]]), np.min(yvals[indices[i]])], color = "black", s = 50, marker = 's');
        axes[i].set_xlabel(str(tuple(labs))+" = "+str(Xpts[indices[i]]));
    axes[0].set_title("Max");
    axes[1].set_title("Min");
    axes[0].set_ylabel("Entanglement prob.");
    plt.show();
    
