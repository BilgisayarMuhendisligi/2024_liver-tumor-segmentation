import random
import numpy as np
import cv2
import pandas as pd
from scipy import ndimage as nd
import pickle
from matplotlib import pyplot as plt
import os
import time
from sklearn.metrics import precision_recall_curve, auc, roc_curve


def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1

    return TP, FP, TN, FN


start_time = time.time()

nums = random.sample(range(1, 50000), 100)

image_dataset = pd.DataFrame()

img_path = "D:/LiTS/CT/"
all_images = sorted(os.listdir(img_path))

for i in range(len(nums)):
    image = all_images[nums[i]]
    print(image)

    df = pd.DataFrame()

    input_img = cv2.imread(img_path + image)

    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("The module works only with grayscale and RGB images!")

    pixel_values = img.reshape(-1)
    df['Pixel_Value'] = pixel_values
    df['Image_Name'] = image

    num = 1
    kernels = []
    for theta in range(2):  # Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  # Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5

                    gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.
                    #                print(gabor_label)
                    ksize = 9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    # Now filter the image and add values to a new column
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  # Labels columns as Gabor1, Gabor2, etc.
                    # print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  # Increment for gabor column label

    edges = cv2.Canny(img, 100, 200)  # Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1  # Add column to original dataframe

    from scipy import ndimage as nd

    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    image_dataset = image_dataset.append(df)

mask_dataset = pd.DataFrame()

mask_path = "D:/LiTS/mask/"
all_masks = sorted(os.listdir(mask_path))

for j in range(len(nums)):
    mask = all_masks[nums[j]]
    print(mask)

    df2 = pd.DataFrame()
    input_mask = cv2.imread(mask_path + mask)

    if input_mask.ndim == 3 and input_mask.shape[-1] == 3:
        label = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)
    elif input_mask.ndim == 2:
        label = input_mask
    else:
        raise Exception("The module works only with grayscale and RGB images!")

    label_values = label.reshape(-1)
    df2['Label_Value'] = label_values
    df2['Mask_Name'] = mask

    mask_dataset = mask_dataset.append(df2)

dataset = pd.concat([image_dataset, mask_dataset], axis=1)

# dataset = dataset[dataset.Label_Value != 0]

X = dataset.drop(labels=["Image_Name", "Mask_Name", "Label_Value"], axis=1)

Y = dataset["Label_Value"].values

from sklearn.preprocessing import LabelEncoder

Y = LabelEncoder().fit_transform(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=50, random_state=42)

results = model.fit(X_train, y_train)

from sklearn import metrics

prediction_test = model.predict(X_test)
##Check accuracy on test dataset.
print("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
print("Precision = ", metrics.precision_score(y_test, prediction_test))
print("F1 = ", metrics.f1_score(y_test, prediction_test))
print("Recall = ", metrics.recall_score(y_test, prediction_test))

end_time = time.time()
time_passed = end_time - start_time
seconds = int(time_passed) % 60
minutes = int(time_passed / 60) % 60
hours = int(time_passed / 3600)

print("Running time:", hours, "hours,", minutes, "minutes,", seconds, "seconds")

TP, FP, TN, FN = perf_measure(y_test, prediction_test)
print("TP, FP, TN, FN =", TP, FP, TN, FN)
calculated_accuracy = (TP + TN) / (TP + TN + FP + FN)
calculated_precision = TP / (TP + FP)
calculated_recall = TP / (TP + FN)
calculated_F1 = (2 * calculated_precision * calculated_recall) / (calculated_precision + calculated_recall)

print("Calculated accuracy =", calculated_accuracy)
print("Calculated precision =", calculated_precision)
print("Calculated F1 =", calculated_F1)
print("Calculated recall =", calculated_recall)

from yellowbrick.classifier import ROCAUC

print("Classes in the image are: ", np.unique(Y))

print("prediction", prediction_test)
print("X test", X_test)
print("Y test", y_test)
print("len", len(X_test))
print("toplam", TN+TP+FN+FP)

##Save the trained model as pickle string to disk for future use
model_name = "rf-model-son"
pickle.dump(model, open(model_name, 'wb'))
