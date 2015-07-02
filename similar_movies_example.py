from __future__ import print_function
import sys
import os
import math
import operator
import itertools
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS

def parse_rating(line):
    """ Parse a line from the ratings dataset
    Args:
        line (str): user_id::movie_id::rating::timestamp
    Returns:
        tuple: (user_id, movie_id, rating)
    """
    items = line.split('::')
    return int(items[0]), int(items[1]), float(items[2])

def parse_movie(line):
    """ Parse a line from the movies dataset
    Args:
        line (str): movie_id::title::genres
    Returns:
        tuple: (movie_id, title)
    """
    items = line.split('::')
    return int(items[0]), items[1]

def dotprod(a, b):
    """ Compute dot product
    Args:
        a (array): vector of values
        b (array): vector of values
    Returns:
        dot_product: dot product of the two input vectors
    """
    return reduce(operator.add, map(lambda k: a[k] * b[k], range(len(a))))

def norm(a):
    """ Compute square root of the dot product
    Args:
        a (array): vector of values
    Returns:
        norm: norm of the vector
    """
    return math.sqrt(dotprod(a, a))

def cossim(a, b):
    """ Compute cosine similarity
    Args:
        a (array): first vector
        b (array): second vector
    Returns:
        cossim: dot product of two normalized vectors
    """
    return dotprod(a, b) / (norm(a) * norm(b))

def rmse(predicted_rdd, actual_rdd):
    """ Root mean squared error between predicted and current
    Args:
        predicted_rdd: predicted ratings (user_id, movie_id, rating)
        actual_rdd: actual ratings (user_id, movie_id, rating)
    Returns:
        rmse (float): root mean squared error
    """
    # Set (user_id, movie_id) as key in predicted_rdd and actual_rdd
    predicted_tuplekey_rdd = predicted_rdd.map(lambda (u, m, r): ((u, m), r))
    actual_tuplekey_rdd    = actual_rdd.map(lambda (u, m, r): ((u, m), r))
    
    # Compute the mean squared error
    mse = (
        predicted_tuplekey_rdd
        .join(actual_tuplekey_rdd)
        .map(lambda a: (a[1][0] - a[1][1]) ** 2)
        .mean()
    )
    # Return the root mean squared error
    return math.sqrt(mse)

def k_fold_cross_validation(
    training_rdd, k=3, ranks=[8], regularizations=[0.1], iterations=5, seed=5L
    ):
    """ Get best parameters (rank, regularization)
        The data set is divided into k subsets, and the model generation is
        repeated k times. Each time, one of the k subsets is used as the test
        set and the other k-1 subsets are put together to form a training set.
        Then the average error across all k trials is computed.
        Finally we choose the parameters of the best model (min average error)
    Args:
        training_rdd: training data
        k: number of folds
        ranks: list of ranks to iterate
        regularizations: list of regularizations to iterate
        iterations: number of iterations in ALS algorithm
        seed: seed for initial randomization in ALS algorithm
    Returns:
        best_rank: rank used in the best model
        best_regularization: regularization used in the best model
    """
    min_error = float('inf')
    best_rank = ranks[0]
    best_regularization = regularizations[0]
    
    # Separate training data in k folds
    fold_weights = [10.0 / k] * k
    training_folds_rdd = training_rdd.randomSplit(fold_weights, seed=0L)
    for fold_rdd in training_folds_rdd:
        fold_rdd.cache()

    # Test all combinations of rank and regularization parameters
    for rank, regularization in itertools.product(ranks, regularizations):
        # cross validation for these parameters
        error_sum = 0
        for i in range(k):
            validation_rdd = training_folds_rdd[i]
            validation_without_ratings_rdd = (
                validation_rdd.map(lambda a: (a[0], a[1]))
            )
            training_group_rdd = (
                sc.union([training_folds_rdd[j] for j in range(k) if j != i])
            )
            model = ALS.train(
                training_group_rdd,
                rank,
                seed=seed,
                iterations=iterations,
                lambda_=regularization
            )
            validation_with_predictions_rdd = model.predictAll(
                validation_without_ratings_rdd
            )
            error_sum += rmse(validation_with_predictions_rdd, validation_rdd)
        # Mean error between all cross validations
        error = error_sum / k
        # Store best parameters
        if error < min_error:
            min_error = error
            best_rank = rank
            best_regularization = regularization
    # Return best parameters
    return best_rank, best_regularization

def similar_items(item_id, amount, model, items):
    """ Compute similar items from a model using cosine similarity
    Args:
        item_id: id of the item to compare
        amount: number of top similarities to get
        model: model obtained by ALS algorithm
        items: RDD of items with name; (id, name)
    Return:
        item_data: (id, name) of the original item
        top_similar_items: (id, name, similarity) of most similar items

    """
    item = (
        model.productFeatures()
        .filter(lambda (id, arr): id == item_id)
        .join(items)
        .map(lambda (id, (arr, title)): (id, title, arr))
        .first()
    )
    item_vector_broadcast = sc.broadcast(item[2])
    
    top_similar_items = (
        model.productFeatures()
        .filter(lambda (id, arr): id != item_id)
        .map(lambda (id, arr): (id, cossim(arr, item_vector_broadcast.value)))
        .join(items)
        .map(lambda (id, (sim, title)): (id, title, sim))
        .takeOrdered(amount, lambda item: -item[2])
    )
    return (item[0], item[1]), top_similar_items

if __name__ == '__main__':
    # Initialize Spark Context
    sc = SparkContext(appName='SimilarMoviesExample')

    # File paths
    ratings_filename = os.path.join('data', 'ratings.dat')
    movies_filename  = os.path.join('data', 'movies.dat')
    
    # Read files
    raw_ratings = sc.textFile(ratings_filename)
    raw_movies  = sc.textFile(movies_filename)
    
    # Parse files to construct RDDs
    ratings_rdd = raw_ratings.map(parse_rating).cache()
    movies_rdd  = raw_movies.map(parse_movie).cache()
    
    # Get training and test datasets from ratings (80/20)
    training_rdd, test_rdd = ratings_rdd.randomSplit([8, 2], seed=0L)
    training_rdd.cache()
    print("Training data: %s" % (training_rdd.count()))

    # Training configuration for k-fold cross-validation
    seed = 5L
    iterations = 5
    regularizations = [0.01, 0.1, 1.0, 10.0]
    ranks = [2, 4, 8, 12]
    k_folds = 5
    
    # Get best parameters with k-fold cross validation
    rank, regularization = k_fold_cross_validation(
        training_rdd, k_folds, ranks, regularizations, iterations, seed
    )
    print("best rank: %s, best regularization: %s" % (rank, regularization))

    # Create model using best parameters 
    model = ALS.train(
        training_rdd,
        rank,
        seed=seed,
        iterations=iterations,
        lambda_=regularization
    )
    
    # Get RMSE using the test dataset
    test_without_ratings_rdd  = test_rdd.map(lambda a: (a[0], a[1]))
    test_with_predictions_rdd = model.predictAll(test_without_ratings_rdd)
    model_error = rmse(test_with_predictions_rdd, test_rdd)
    print('RMSE: %s' % model_error)

    # Test movie similarity
    movie_ids = [
        260,  # Star Wars: Episode IV - A New Hope (a.k.a. Star Wars) (1977)
        32,   # 12 Monkeys
        589,  # Terminator 2: Judgment Day (1991)
        6874, # Kill Bill: Vol. 1 (2003)
        8957, # Saw (2004)
        648,  # Mission: Impossible (1996)
        4246  # Bridget Jones's Diary (2001)
    ]
    # Print similarity lists in 'result.txt' file
    output = open('result.txt', 'w+')
    for movie_id in movie_ids:
        movie, similar_movies = similar_items(movie_id, 15, model, movies_rdd)
        print("Similar movies to: %s" % movie[1].encode('utf-8'), file=output)
        for i in range(len(similar_movies)):
            print(
                "%2s) [%0.2f] %s" % (
                    i + 1,
                    similar_movies[i][2],
                    similar_movies[i][1].encode('utf-8')
                ),
                file=output
            )
        print('', file=output)

    # Stop the Spark Context
    sc.stop()