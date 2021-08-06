# Speech-driven Hand Gesture Generation Demo
This repository can be used to reproduce our results of applying our model to the English dataset.

If you want to learn more about the model - this [video](https://youtu.be/Iv7UBe92zrw) is a good start.

Example of generate motion can be seen in the [demo video](https://youtu.be/tQLVyTVtsSU).

## Requirements
* python 3
* ffmpeg (to visualize the results)

## Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
```
./generate.sh  data/audio*.wav
```
Where in place of `audio*.wav` you can use any file from the folder `data`, which are chunks of the test sequences.
 Alternatively, you can download more audios for testing from [the Trinity Speech-Gesture dataset](https://trinityspeechgesture.scss.tcd.ie/Audio/).
(The recordings 'NaturalTalking_01.wav' and 'NaturalTalking_02.wav' were not used in training and were left them for testing)

## Training on your own data
For training on your own data we refer you to the [original repository](https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder) with the official implementation of the paper.

## Citation
Here is the citation of our paper in bib format:
```
@article{kucherenko2021moving,
author = {Taras Kucherenko and Dai Hasegawa and Naoshi Kaneko and Gustav Eje Henter and Hedvig Kjellström},
title = {Moving Fast and Slow: Analysis of Representations and Post-Processing in Speech-Driven Automatic Gesture Generation},
journal = {International Journal of Human–Computer Interaction},
volume = {37},
number = {14},
pages = {1300-1316},
year  = {2021},
publisher = {Taylor & Francis},
doi = {10.1080/10447318.2021.1883883},
URL = {https://doi.org/10.1080/10447318.2021.1883883},
eprint = {https://doi.org/10.1080/10447318.2021.1883883}
}
```

If you are going to use Trinity Speech-Gesture dataset, please don't forget to cite them as described in [their website](https://trinityspeechgesture.scss.tcd.ie)

## Contact
If you encounter any problems/bugs/issues please contact me on Github or by emailing me at tarask@kth.se for any bug reports/questions/suggestions. I prefer questions and bug reports on Github as that provides visibility to others who might be encountering same issues or who have the same questions.
