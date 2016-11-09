import pyaudio
import numpy as np
from detector import load_pipeline

_osx_mic_name = u'Built-in Microph'
_format = pyaudio.paInt16
_channels = 2
_rate = 44100

def get_device_number(pa, dev_name=_osx_mic_name):
    for k in range(pa.get_device_count()):
	if pa.get_device_info_by_index(k)['name'] == dev_name:
	    return k
    return -1

def open_mic_stream(block_time=0.5, dev_name=_osx_mic_name):
    input_block_sz = int(block_time*_rate)
    pa = pyaudio.PyAudio()
    mic_dev_idx = get_device_number(pa, dev_name)
    stream = pa.open(
	    format=_format,
	    channels=_channels,
	    rate=_rate,
	    input=True,
	    input_device_index=mic_dev_idx,
	    frames_per_buffer=input_block_sz)
    return stream, pa

def get_sample(stream, buf_time=0.5):
    return stream.read(int(buf_time*_rate), exception_on_overflow=False)

def convert_sample_to_numpy(sample, samp_format=_format):
    dtype_map = {
	    pyaudio.paInt16: np.int16,
	    pyaudio.paFloat32: np.float32}
    return np.fromstring(sample, dtype=dtype_map[samp_format])

def get_np_samp(stream, buf_time=0.49, samp_format=_format):
    samp = get_sample(stream, buf_time)
    np_samp = convert_sample_to_numpy(samp, samp_format)
    return np_samp

def do_power_loop(stream, n_iterations):
    for it in range(n_iterations):
	np_samp = get_np_samp(stream)
	print(np.sqrt(np.mean(np_samp**2)))

def do_coffe_loop(stream, n_iterations):
    pipeline = load_pipeline()
    for it in range(n_iterations):
	np_samp = get_np_samp(stream)
	print(pipeline.predict(np_samp))
