## objdetect.py
    Used Google Mediapipe API (computer vision API model) and general object detection and computer vision methods to detect pose estimation (position of arms, wrists, etc). Analyzed pose estimation landmarks 
    and defined thresholds for certain stages of the pull-up to accurately detect number of repetitions and the different phases of the exercise. Defined functions to calculate angles, analyze frames in video, 
    and start and stop video capture. Used these thresholds to label the different phases of the exercise and number of repetitions for our RNN.

## rnn.py
    Define a recurrent neural network to detect patterns in pull up form, using the labels from objdetect.py to detect various lapses in form or number of pull-up repetitions in newer videos that the user inputs.
    Uses transfer learning techniques by using the Mediapipe API to handle feature extraction, while the RNN focuses on recognizing sequences of these pre-labeled poses to improve accuracy in detect repetitions and 
    stages of a pull-up in the user's video input.

## API
I'm gonna make an API so I can make a web app that doesn't have to access the back end; it will be a Rest API (POST, GET, etc.)
