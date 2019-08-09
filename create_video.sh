#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: create_video audio_file gesture_file"
    exit 1
fi

# Set params
input_audio=$1
input_gesture_file=$2
output_video='result/Gesture_Video.mp4'

# Generate video for gestures from the 3d position
python visualization/model_animator.py --input=$input_gesture_file --out='data/temp_gesture_video.mp4'

# Add audio to the video
ffmpeg -i 'data/temp_gesture_video.mp4' -i $input_audio -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k $output_video

