import numpy as np
import scipy
from scipy.signal import butter, lfilter, lfilter_zi

# PLEASE NOTE SOME PARTS ARE CODED IN A C-STYLE WAY FOR EASIER PORTING LATER


def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass', output='ba')
    return b, a


def butter_lowpass_filter(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, _ = lfilter(b, a, data, zi = zi*data[0])
    return y

def generate_features(draw_graphs, raw_data, axes, sampling_freq, scale_axes, center_data, filter_order, cutoff):
    
    # features is a 1D array, reshape so we have a matrix with shape (n_timesteps, n_features)
    n_timesteps = len(raw_data) // len(axes)
    n_features = len(axes)
    raw_data = raw_data.reshape(n_timesteps, n_features) # (timesteps, features)

    features = []
    centered_graph = {}
    smoothed_graph = {}

    # Get each feature data. Shape (timesteps,)
    # We could have done some steps directly with proper indexing, although I want to keep the code
    # easy to port
    for ax, fx in enumerate(raw_data.T):

        #1 Subtract mean
        if center_data:
            fx = fx - fx.mean()

            # Graphing for edge impulse
            if (draw_graphs):
                centered_graph[axes[ax]] = list(fx)

        # butterworth low pass filter
        # Pythonic way
        fx = butter_lowpass_filter(fx, cutoff, sampling_freq, filter_order)

        # C style
        # TODO calculate z transform and get coefficients (for a given cutoff f)
        # then discretize

        #3 Scaling (optional)
        fx = scale_axes * fx

        # we save bandwidth by only drawing graphs when needed
        if (draw_graphs):
            smoothed_graph[axes[ax]] = list(fx)

        # we need to return a 1D array again, so flatten here again
        features.append(fx)

    features_merged = np.vstack(features).T.flatten()

    # draw the graph with time in the window on the Y axis, and the values on the X axes
    # note that the 'suggestedYMin/suggestedYMax' names are incorrect, they describe
    # the min/max of the X axis
    graphs = []
    if (draw_graphs):
        graphs.append({
            'name': 'Output signal',
            'X': smoothed_graph,
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': -2,
            'suggestedYMax': 2
        })

        if (center_data):
            graphs.append({
            'name': 'Centered signal',
            'X': centered_graph,
            'y': np.linspace(0.0, raw_data.shape[0] * (1 / sampling_freq) * 1000, raw_data.shape[0] + 1).tolist(),
            'suggestedYMin': -2,
            'suggestedYMax': 2
            })


    return { 'features': features_merged, 'graphs': graphs }
