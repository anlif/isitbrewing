import matplotlib.pyplot as plt
import numpy as np
from pyAudioAnalysis import audioFeatureExtraction 
from sklearn import preprocessing, svm, linear_model
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from scipy.signal import hamming

pipeline_joblib_fname = 'kaffe_pipeline.gz'

class WindowedExtractor(object):
    def __init__(self, Fs, win_size_sec=0.5):
	self.win_size_sec = win_size_sec
        self.Fs = Fs
        self.window_size = int(win_size_sec*Fs)
    
    def fit(self, X, y):
        return self
    
    def transform(self, signal):
	n_wins = signal.shape[0]/self.window_size
	windows = np.array_split(signal[0:self.window_size*n_wins], n_wins)
	f_extract = lambda s: audioFeatureExtraction.stFeatureExtraction(
		s*hamming(s.shape[0]),
		self.Fs,
		self.window_size,
		self.window_size)
	feature_list = map(f_extract, windows)
	feature_matrix = reduce(lambda f1, f2: np.vstack((f1,f2.T)), feature_list[1:], feature_list[0].T)
	return feature_matrix[:,0:22]

def build_pipeline(raw_extractor):
    scaler = preprocessing.StandardScaler()
    classifier = svm.SVC(kernel='rbf', probability=False)
    feature_selector = RFECV(
	    estimator=classifier, 
	    step=1, 
	    cv=StratifiedKFold(2), 
	    scoring='accuracy')
    kaffe_pipeline = Pipeline(
	    [('raw', raw_extractor),
	     ('scaler', scaler),
	     #('feature_select', feature_selector),
	     ('classify', classifier)])
    return kaffe_pipeline

def get_labels(X, raw_extractor, t_list):
    y = np.repeat(-1, X.shape[0])
    for (t_start, t_end) in t_list:
	idx_start = int(t_start/(float(raw_extractor.window_size)/raw_extractor.Fs))
	idx_end = idx_start + int((t_end-t_start)/(float(raw_extractor.window_size)/raw_extractor.Fs))
	all_idx = np.arange(X.shape[0])
	brew_idx = np.logical_and(all_idx > idx_start, all_idx < idx_end)
	y[brew_idx] = 1
    return y

def save_pipeline(pipeline):
    joblib.dump(pipeline, pipeline_joblib_fname)

def load_pipeline():
    return joblib.load(pipeline_joblib_fname) 
