# Time Sequence Prediction
This is a toy example for beginners to start with both pytorch and time sequence prediction. Two LSTMCell is used in this example to learn some sine wave signals starting at different phase. After learning the sine waves, the network try to predict the signals in the future. The results is shown in the picture below.

## Usage

```
python generate_sine_wave.py
python train.py
```

## Result
The initial signal and the prediction results are shown in the image. We firstly give some initial signals (full line). The network will  subsequently give some predicted results (dash line). It can be concluded that the network can generate new sine waves.
![image](https://cloud.githubusercontent.com/assets/1419566/24184438/e24f5280-0f08-11e7-8f8b-4d972b527a81.png)
