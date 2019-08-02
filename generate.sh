#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./generate.sh audio_file"
    exit 1
fi


input_audio=$1

echo "Encoding audio ..."
python -W ignore encode_audio.py --input_audio=$input_audio --output_file=data/encoded_audio.npy
echo "Predicting motion encoding ..."
python -W ignore predict.py models/Model_MFCC_Pros_Best.hdf5 data/encoded_audio.npy data/encoded_motion.txt
echo "Decoding motion ..."
python -W ignore decode_motion.py
echo "Creating a video with this motion and audio ..."
./create_video.sh $input_audio result/gestures.txt

echo "The gestures 3D coordinates were written in the file 'result/gestures.txt'"
echo "The video with the produced gestures was written in the file 'Gesture_Video.mp4'"
