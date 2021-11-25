# Melakarta (Raga) Detecion
Raga, which is a melodic framework, is one of the core concepts in Carnatic music. Determination of Raga plays a critical role
in information retrieval concerning Carnatic songs. Each Raga plays a significant role in bringing out
different emotions in the listener. When there is a need to create a particular atmosphere, it could be
advantageous to choose Carnatic songs of a specific Raga. Consequently, Raga classification has significant
utility. Transfer learning and audio processing are the two different approaches experimented in this project to detect Melakarta
Raga. In transfer learning, knowledge is transferred from a variety of base tasks - instrument detection,
spoken digit classification and genre classification to the Melakarta Raga detection model, thus exploring
different ways to choose the optimal base task for this transfer learning problem. The audio processing
approach uses individual note extraction and vocal range detection to obtain the set of Swaras used in the
audio sample. Classification of Melakarta involves comparing the list of Swaras in the audio sample to the
dataset comprising the combination of Swaras for each Melakarta.


## How to run the code?
Install the requirements<br>
```
pip install -r requirements.txt
```


### Transfer Learning
1. Create MFCC datasets, for base task and transfer task, by placing all audio files in a folder. All audio files should be named in the format `<classification>_<file_no>.wav`
   ```
   python create_csv.py <audio_folder> <input_variable_filename> <output_variable_filename>
   ```
   For example,
   ```
   python create_csv.py data/music baseX.csv baseY.csv
   ```

2. Run transfer learning script by passing the audio file to be predicted (.wav) and the data files for base task and target task.<br>

    ```
    python transfer_learning.py <base_task_input> <base_task_output> <transfer_task_input> <transfer_task_output> <test_audio_file>
    ```

    For example,
    ```
    python transfer_learning.py baseX.csv baseY.csv transferX.csv transferY.csv test.wav
    ```

### Audio Processing
Run the audio processing script by executing the below command
```
python audio_processing.py <test_audio_file>
```
## Experiment Results
Both the methods were tested aginst a manually generated dataset of 3 Ragas - Dheerasankarabharanam, Harikamboji and Shanmugapriya.
Distribution of the Raga dataset:
<img src="https://user-images.githubusercontent.com/47625221/143379985-c1803bb5-ae5a-4988-a86e-c81aafff7838.png" width="500"/>

Audio processing algorithm was tested with 4 different types of audio samples - instrumental, vocal, vocal with background, vocal after removing background<br>
Experiment results:
<img src="https://user-images.githubusercontent.com/47625221/143380533-c6d8f2a1-27aa-4ec9-a44c-d0306214010b.png" width="500"/>

Transfer learning model was trainined with 3 different base tasks - nstrument detection, spoken digit classification and genre classification<br>
Experiment results:
<img src="https://user-images.githubusercontent.com/47625221/143380799-629d2e97-01cf-456e-9cc9-d041532597bd.png" width="500"/>

More details on the algorithm, experiment and result analysis can be found here.
