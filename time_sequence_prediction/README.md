# Time Sequence Prediction by pytorch
This is an simple and illustrative example showing how to model the time sequence with LSTM(Long Short-Term Memory) . Several sine waves are input into the model and the modle learns how to generate the future waves according to given states. The result is as following.
![image](https://cloud.githubusercontent.com/assets/1419566/23689065/1d6e9900-03f3-11e7-958b-80066f2e9472.png)

## Usage 
1. Generate the training data
```
python generate_sine_wave.py
```

2. Train the model and predict the future states
```
python train.py
```

## The model
Stacked LSTM nodes are used to learn the patterns of the input signal.
