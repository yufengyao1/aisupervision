# calculate filterbank features. Provides e.g. fbank and mfcc features for use in ASR applications
# Author: James Lyons 2012
import time
import numpy
# from numba import jit
# from numba import njit
from fbank import sigproc


def logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
             nfilt=40, nfft=512, lowfreq=64, highfreq=None, dither=1.0, remove_dc_offset=True, preemph=0.97, wintype='hamming'):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, dither, remove_dc_offset, preemph, wintype)
    log_feat = numpy.log(feat)
    return log_feat


# @njit
def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 1127.0 * numpy.log(1+hz/700.0)


# @njit
def get_filterbanks(nfilt=80, nfft=512, samplerate=16000, lowfreq=20, highfreq=None):  # 固定值
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate/2
    # assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2" #删

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)

    # check kaldi/src/feat/Mel-computations.h
    fbank = numpy.zeros((nfilt, nfft//2+1))
    mel_freq_delta = (highmel-lowmel)/(nfilt+1)
    for j in range(0, nfilt):
        leftmel = lowmel+j*mel_freq_delta
        centermel = lowmel+(j+1)*mel_freq_delta
        rightmel = lowmel+(j+2)*mel_freq_delta
        for i in range(0, nfft//2):
            mel = hz2mel(i*samplerate/nfft)
            if mel > leftmel and mel < rightmel:
                if mel < centermel:
                    fbank[j, i] = (mel-leftmel)/(centermel-leftmel)
                else:
                    fbank[j, i] = (rightmel-mel)/(rightmel-centermel)
    return fbank


fb = get_filterbanks()


def fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
          nfilt=40, nfft=512, lowfreq=0, highfreq=None, dither=1.0, remove_dc_offset=True, preemph=0.97,
          wintype='hamming'):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
     winfunc=lambda x:numpy.ones((x,))   
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    # highfreq = highfreq or samplerate/2
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, dither, preemph, remove_dc_offset, wintype)
    pspec = sigproc.powspec(frames, nfft)  # nearly the same until this part
    # fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq) #常量，计算一次
    feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)  # if feat is zero, we get problems with log
    return feat
