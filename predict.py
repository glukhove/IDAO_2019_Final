import datetime
import numpy as np
import pickle
import ast
import sys
import catboost
from numba import jit


MODELS_FILE = 'models.pkl'
OUTPUT_HEADER = 'datetime,target_{},target_{},target_{},target_{},target_{}'

HOUR_IN_MINUTES = 60
DAY_IN_MINUTES = 24 * HOUR_IN_MINUTES
WEEK_IN_MINUTES = 7 * DAY_IN_MINUTES

SHIFTS = [3, 5, 7, 10, 15, 20,
    HOUR_IN_MINUTES // 2,
    HOUR_IN_MINUTES,
    HOUR_IN_MINUTES*2,
    DAY_IN_MINUTES,
    DAY_IN_MINUTES * 2,
    WEEK_IN_MINUTES,
    WEEK_IN_MINUTES * 2]
WINDOWS = [3, 5, 7, 10, 15, 20,
    HOUR_IN_MINUTES // 2,
    HOUR_IN_MINUTES,
    HOUR_IN_MINUTES*2,
    DAY_IN_MINUTES,
    DAY_IN_MINUTES * 2,
    WEEK_IN_MINUTES,
    WEEK_IN_MINUTES * 2]


# def np_exponential_evarage(x, w):
#     result = np.convolve(x, np.ones(w), 'same')/w
#     return result

@jit()
def exponential_smoothing_numba(series, alpha):
    """
        series - dataset with timestamps
        alpha - float [0.0, 1.0], smoothing parameter
    """
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def extractor(dt, history, parameters):
    features = []

    for shift in SHIFTS:
        for window in WINDOWS:
            if window > shift:
                continue
            if window == shift:
                features.append(sum(history[-shift:]))
            else:
                features.append(sum(history[-shift:-shift + window]))
                
    
    week = [0]*7
    week[dt.weekday()] = 1
    features.extend(week)
    
    hour = dt.hour
    minute = dt.minute


    # encode hour with sin/cos transformation
    # credits - https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
    features.append(np.sin(2*np.pi*hour/24))  
    features.append(np.cos(2*np.pi*hour/24))
    features.append(hour*60+minute)
    features.append(np.sin(2*np.pi*minute/24*60))
    features.append(np.cos(2*np.pi*minute/24*60))
    

    return np.array(features)


if __name__ == '__main__':
    models = pickle.load(open(MODELS_FILE, 'rb'))

    input_header = input()
    output_header = OUTPUT_HEADER.format(*sorted(list(models['models'].keys())))
    print(output_header)
    
    all_features = []
    all_queries = []
    while True:
        # read data, calculate features line by line for memory efficient
        try:
            raw_line = input()
        except EOFError:
            break
                    
        line = raw_line.split(',', 1)
        dt = datetime.datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S')
        history = list(map(int, line[1][2:-2].split(', ')))
        history = exponential_smoothing_numba(history, 0.3)
        features = extractor(dt, history, None)
        
        all_features.append(features)
        all_queries.append(line[0])
    
    # predict all objects for time efficient
    predictions = []
    for position, model in models['models'].items():
        predictions.append(model.predict(all_features))
    
    for i in range(len(predictions[0])):
        print(','.join([all_queries[i]] + list(map(lambda x: str(x[i]), predictions))))
