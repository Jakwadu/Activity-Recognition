import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from scipy.signal import butter, filtfilt
from matplotlib import pyplot

import tensorflow as tf

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Add, LeakyReLU
from keras.layers import Concatenate, BatchNormalization, LSTM
from keras.metrics import categorical_crossentropy
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

from tensorflow import convert_to_tensor

# Dictionary for mapping activity code to description
activity_translator = {
                    1 : 'Working at Computer',
                    2 : 'Stand, Walk, Use Stairs',
                    3 : 'Standing',
                    4 : 'Walking',
                    5 : 'Using Stairs',
                    6 : 'Walking and Talking',
                    7 : 'Standing and Talking'}

filePath = os.path.dirname(os.path.realpath(__file__))

fileList = os.listdir(filePath)

dataList = list()
encoder = LabelBinarizer()

sampleWindow = 20
nSensors = 3

# Read accelerometer data. The sensors are uncalibrated, so it's best to scale the data 
# from each paticipant separately. Each dimension from a given sensor readings will also 
# be scaled individually for the same reason.

for item in fileList:
    if item[-4:] == '.csv':
        tmp = pd.read_csv(item, header=None)
        scaler = StandardScaler()
        for column in tmp.columns[1:-1]:
            tmp[column] = tmp[column].astype('float')
            tmp[column] = scaler.fit_transform(tmp[column].values.reshape(-1,1))
        tmpData = tmp.values[:,1:]
        # Delete activities coded "0"
        acts = tmpData[:,-1]
        idx1 = np.where(acts==0)
        tmpData = np.delete(tmpData, idx1, axis=0)
        # Binarise activities
        encodedLabels = encoder.fit_transform(tmpData[:,-1].reshape(-1,1))
        processedData = np.hstack((tmpData[:,:-1], encodedLabels))
        dataList.append(processedData)
cl = encoder.classes_.astype('int')
person1 = dataList[0]
activity = encoder.inverse_transform(person1[:,-7:])

# pyplot.figure(1)
# # Setup axes
# pyplot.plot(person1[:,0])
# pyplot.plot(person1[:,1])
# pyplot.plot(person1[:,2])
# pyplot.xlabel('Sample Number')
# pyplot.ylabel('Scaled Acceleration')
# pyplot.legend(['x','y','z'])
# pyplot.twinx()
# # Plot sensor readings and activity states
# pyplot.plot(activity, color='tab:red')
# pyplot.ylabel('Activity')
# pyplot.title('Accelerometer Readings with Activity')
# pyplot.legend(['Activity'])
# pyplot.tight_layout()

# pyplot.show()

# Explore shape of training data per person
psn_no = 1
for psn in dataList:  
    print("Person ",psn_no,": ",psn.shape)
    psn_no += 1

# Stack data to facilitate model use
fullData_temp = np.concatenate(dataList, axis = 0)
dataHeight = np.shape(fullData_temp)[0]
dataTail = dataHeight % sampleWindow
fullData = fullData_temp[:-dataTail]
X = fullData[:,:nSensors]
y = fullData[:,nSensors:]

# Set up low pass filter
b, a = butter(5, 0.5, fs=20)
filterOut = np.array([filtfilt(b, a, sig) for sig in np.transpose(X)])
X = np.transpose(filterOut)

# Transpose  feature data to the correct orientation
print("Shape of X: ", X.shape)
shapedX = np.reshape(X,(sampleWindow,nSensors,-1))
print("New shape of X: ", shapedX.shape)
rotatedX = np.transpose(shapedX, (2,0,1))
print("Shape of X for model input: ", rotatedX.shape)

# Create indices for sample window 
idx2 =  [x*sampleWindow for x in range(0,np.shape(fullData)[0]//sampleWindow)]
# Find the most frequently occuring activity within the window and assign as the target output
ySlicesTemp = np.vsplit(y, idx2)
ySlices = list()
for cat in ySlicesTemp:
    temp = np.mean(cat, axis=0)
    maxIdx = np.where(temp==np.max(temp))
    temp2 = np.zeros((1,7), dtype=int)
    temp2[0,maxIdx] = 1
    ySlices.append(temp2)
yShaped = np.concatenate(ySlices, axis=0)[:-1]
# Print number of X and y instances to check consistency
print('Number of X slices: ', rotatedX.shape[0])
print('Number of y encodings: ', yShaped.shape[0])
# Shuffle data to remove minimise bias
X_,y_ = shuffle(rotatedX,yShaped)
# Create train test split of 60% to 40%
train_cv_X, test_X, train_cv_y, test_y = train_test_split(X_, y_, test_size=0.3, random_state=42)

# Enable Tensorboard to monitor training progress and observe model architecture
tBoard = TensorBoard(log_dir='./TensorBoard', histogram_freq=0, write_graph=True, write_images=True)

input_ = Input((sampleWindow, nSensors))

lstm = LSTM(300)(input_)

dense1 = Dense(800, activation='relu')(lstm)
dense1 = Dense(400, activation='relu')(dense1)

output_ = Dense(7, activation='softmax')(dense1)

clf = Model(inputs=input_, outputs=output_)

print(clf.summary())
clf.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks for reducing learning rate and early stopping
reduce_lr = ReduceLROnPlateau(patience=2, factor=0.2, min_lr=1.6e-6, monitor='val_loss')
stopEarly = EarlyStopping(patience=10, monitor='val_loss')

# Fit model to training data
clf.fit(train_cv_X, train_cv_y, epochs=50, batch_size=32, callbacks=[reduce_lr, stopEarly], validation_data=(test_X, test_y))

# Save model to current directory
clf.save('activityDetector.h5f')

# Load model from current directory
clf = load_model('activityDetector.h5f')

# Generate predictions for test inputs
raw_prediction_ = clf.predict(test_X)

# Binarise predictions paded on the category with the highest output
pred_y = np.zeros(np.shape(raw_prediction_), dtype=int)

for iter_ in range(np.shape(raw_prediction_)[0]):
    idx = np.where(raw_prediction_[iter_]==np.max(raw_prediction_[iter_]))
    pred_y[iter_,idx] = 1

# Generate classification report covering precision, recall and F1 score
print(classification_report(test_y, pred_y, target_names=activity_translator.values()))

# De-binarise predictions and true activities to enable plotting
predActivity = encoder.inverse_transform(pred_y)
trueActivity = encoder.inverse_transform(test_y)
# Create normalised confusion matrix
confusion_ = confusion_matrix(trueActivity, predActivity, labels=encoder.classes_.astype('int'))
norm_confusion_ = confusion_.astype('float')/confusion_.sum(axis=1)[:, np.newaxis]

# Create plot labels
tickLabels = list()
for act in cl:
    tmp_ = activity_translator[act]
    tickLabels.append(str(tmp_))
# Plot confusion matrix
pyplot.figure(2)
pyplot.title('Normalised Confusion Matrix')
pyplot.imshow(norm_confusion_)
pyplot.xlabel('Predicted Activity')
pyplot.ylabel('True Activity')
pyplot.xticks(ticks=range(7), labels=tickLabels, rotation=45, ha='right', rotation_mode='anchor')
pyplot.yticks(ticks=range(7), labels=tickLabels)
pyplot.colorbar()
pyplot.tight_layout()
pyplot.show()