# Speech-driven Hand Gesture Generation Demo
This repository can be used to reproduce our results of applying our model to a new English dataset.

If you want to learn more about the model - this [video](https://youtu.be/Iv7UBe92zrw) is a good start.
For more information about the dataset used we refer to the Trinity Speech-Gesture dataset [website](https://trinityspeechgesture.scss.tcd.ie/)

## Requirements
*  python 3

## Installation
```
pip install -r requirements.txt
```

## Usage
```
./generate.sh <audio_file>
```
Where in place of `<audio_file>` you can use any file from the folder `data` or download more [audios](https://trinityspeechgesture.scss.tcd.ie/Audio/) for testing from the dataset.
(We did not use recordings 'NaturalTalking_01.wav' and 'NaturalTalking_02.wav' during training and left them for testing)

## Training on your own data
For training on your own data we refer you to the [original repository](https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder) with the official implementation of the paper.

## Citation
Here is the citation of our paper in bib format:
```
@inproceedings{kucherenko2019analyzing,
  title={Analyzing Input and Output Representations for Speech-Driven Gesture Generation},
  author={Kucherenko, Taras and Hasegawa, Dai and Henter, Gustav Eje  and Kaneko, Naoshi and Kjellstr{\"o}m, Hedvig},
  booktitle=={International Conference on Intelligent Virtual Agents (IVA ’19)},
  year={2019},
  publisher = {ACM},
}
```

If you are going to use Trinity Speech-Gesture dataset, please don't forget to cite them as described in [their website](https://trinityspeechgesture.scss.tcd.ie/Readme.txt)

## Contact
If you encounter any problems/bugs/issues please contact me on Github or by emailing me at tarask@kth.se for any bug reports/questions/suggestions. I prefer questions and bug reports on Github as that provides visibility to others who might be encountering same issues or who have the same questions.
