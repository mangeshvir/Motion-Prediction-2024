def select_features():
    """
    Select features for the model.
    
    Returns
    -------
    tuple
        A tuple containing a list of selected features and the number of features.
    """
    features = [
        "trackId",
        "xCenter",
        "yCenter",
        "heading",
        "xVelocity",
        "yVelocity",
        "xAcceleration",
        "yAcceleration",
        "lonVelocity",
        "latVelocity",
        "lonAcceleration",
        "latAcceleration",
    ]
    number_of_features = len(features)
    return features, number_of_features
