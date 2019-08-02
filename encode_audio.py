import sys,os
sys.path.insert(1, os.path.join(sys.path[0], 'helpers'))
import numpy as np
import argparse

from tools import *

DATA_DIR = ''
N_CONTEXT = 60  # Number of context: Total of how many pieces are seen before and after, when it is 60, 30 before and after
FEATURES = "MFCC+Pros"
N_INPUT = 30 # Total number of features
SILENCE_PATH = "helpers/silence.wav"


def pad_sequence(input_vectors):
    """
    Pad array of features in order to be able to take context at each time-frame
    We pad N_CONTEXT / 2 frames before and after the signal by the features of the silence
    Args:
        input_vectors:      feature vectors for an audio

    Returns:
        new_input_vectors:  padded feature vectors
    """

    if FEATURES == "MFCC":

        # Pad sequence not with zeros but with MFCC of the silence

        silence_vectors = calculate_mfcc(SILENCE_PATH)
        mfcc_empty_vector = silence_vectors[0]

        empty_vectors = np.array([mfcc_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "Pros":

        # Pad sequence with zeros

        prosodic_empty_vector =[0, 0, 0, 0]

        empty_vectors = np.array([prosodic_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "MFCC+Pros":

        silence_vectors = calculate_mfcc(SILENCE_PATH) #
        mfcc_empty_vector = silence_vectors[0]

        prosodic_empty_vector = [0, 0, 0, 0]

        combined_empty_vector = np.concatenate((mfcc_empty_vector, prosodic_empty_vector))

        empty_vectors = np.array([combined_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "Spectro":

        silence_spectro = calculate_spectrogram(SILENCE_PATH)
        spectro_empty_vector = silence_spectro[0]

        empty_vectors = np.array([spectro_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "Spectro+Pros":

        silence_spectro = calculate_spectrogram(SILENCE_PATH)
        spectro_empty_vector = silence_spectro[0]

        prosodic_empty_vector = [0, 0, 0, 0]

        combined_empty_vector = np.concatenate((spectro_empty_vector, prosodic_empty_vector))

        empty_vectors = np.array([combined_empty_vector] * int(N_CONTEXT / 2))

    # append N_CONTEXT/2 "empty" mfcc vectors to past
    new_input_vectors = np.append(empty_vectors, input_vectors, axis=0)
    # append N_CONTEXT/2 "empty" mfcc vectors to future
    new_input_vectors = np.append(new_input_vectors, empty_vectors, axis=0)

    return new_input_vectors


def create_vectors(audio_filename):
    """
    Extract features from a given pair of audio and motion files
    Args:
        audio_filename:    file name for an audio file (.wav)

    Returns:
        input_with_context   : speech features
        output_with_context  : motion features
    """

    visualize=False

    # Step 1: Vactorizing speech, with features of N_INPUT dimension, time steps of 0.01s
    # and window length with 0.025s => results in an array of 100 x N_INPUT

    if FEATURES == "MFCC":

        input_vectors = calculate_mfcc(audio_filename)

    elif FEATURES == "Pros":

        input_vectors = extract_prosodic_features(audio_filename)

    elif FEATURES == "MFCC+Pros":

        mfcc_vectors = calculate_mfcc(audio_filename)

        pros_vectors = extract_prosodic_features(audio_filename)

        mfcc_vectors, pros_vectors = shorten(mfcc_vectors, pros_vectors)

        input_vectors = np.concatenate((mfcc_vectors, pros_vectors), axis=1)

    elif FEATURES =="Spectro":

        input_vectors = calculate_spectrogram(audio_filename)

    elif FEATURES == "Spectro+Pros":
        spectr_vectors = calculate_spectrogram(audio_filename)

        pros_vectors = extract_prosodic_features(audio_filename)

        spectr_vectors, pros_vectors = shorten(spectr_vectors, pros_vectors)

        input_vectors = np.concatenate((spectr_vectors, pros_vectors), axis=1)



    # Step 4: Retrieve N_CONTEXT each time, stride one by one
    input_with_context = np.array([])

    strides = len(input_vectors)

    input_vectors = pad_sequence(input_vectors)

    for i in range(strides):
        stride = i + int(N_CONTEXT/2)
        if i == 0:
            input_with_context = input_vectors[stride - int(N_CONTEXT/2) : stride + int(N_CONTEXT/2) + 1].reshape(1, N_CONTEXT+1, N_INPUT)
        else:
            input_with_context = np.append(input_with_context, input_vectors[stride - int(N_CONTEXT/2) : stride + int(N_CONTEXT/2) + 1].reshape(1, N_CONTEXT+1, N_INPUT), axis=0)

    return input_with_context


def encode_and_save(audio_input_file, output_file):

    audio_encoded = create_vectors(audio_input_file)
    np.save(output_file, audio_encoded)


if __name__ == '__main__':
    # Parse command line params

    parser = argparse.ArgumentParser(
        description='Encode an audio file and save it into numpy format')
    parser.add_argument('--input_audio', '-i', default='data/audio_segment.wav',
                        help='Path to the input file with the motion')
    parser.add_argument('--output_file', '-o', default='data/audio_encoded.npy',
                        help='Path to the output file with the video')
    args = parser.parse_args()

    encode_and_save(args.input_audio, args.output_file)
