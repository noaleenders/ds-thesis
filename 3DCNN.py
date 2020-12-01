import glob
from tensorflow import keras
import cv2
import numpy as np
import random
import scipy
from scipy.ndimage.filters import gaussian_filter
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split


class VidDataGenerator(Sequence):

    def __init__(self, video_paths, batch_size=1, n_classes=4, shuffle=True, to_fit=True, aug=True):
        self.video_paths = video_paths
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.aug = aug

        self.on_epoch_end()

    def __len__(self):
        "Return the number of batches per epoch"
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        "Generate a batch of data"

        # get indices for the batch for the data
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # get video file name from indices
        vid_files_temp = [self.video_paths[i] for i in indices]

        # generate data from the indices
        X = self.generate_x(vid_files_temp)

        if self.to_fit:
            # get corresponding labels
            y = self.generate_y(vid_files_temp)
            return X, y

        else:
            return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.video_paths))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def generate_x(self, vid_files_temp):
        "Read video data"

        X_data = []

        for file in vid_files_temp:
            frames_list = []
            vidcap = cv2.VideoCapture(file)

            # get number of frames
            frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

            # read frames and add to array
            fc = 0
            success = True
            while (fc < frameCount and success):
                success, image = vidcap.read()

                image = cv2.resize(image, (200, 95))

                # normalize the frames
                image = image.astype('float32')
                image /= 255.0

                frames_list.append(image)
                fc += 1

            # only get the last 10 frames
            if len(frames_list) > 10:
                frames_list = frames_list[-10:]

            X_data.append(frames_list)

        # pad videos in the batch to 10 frames if less than 10
        X_data_padded = pad_sequences(X_data, padding="post", dtype="float32", maxlen=10)

        # getting random numbers for augmentation
        r_number = random.uniform(0, 1)

        if self.aug:
            # augmentation

            if r_number < 0.5:
                # vertically flip batch of videos
                X_data_padded = flip_batch(X_data_padded)

        return X_data_padded

    def generate_y(self, vid_files_temp):
        "Get labels from the data"

        y = []
        for file in vid_files_temp:
            one_hot = [0] * 4
            if 'handShake' in file:
                one_hot[0] = 1
            if 'highFive' in file:
                one_hot[1] = 1
            if 'hug' in file:
                one_hot[2] = 1
            if 'kiss' in file:
                one_hot[3] = 1
            y.append(one_hot)
        return np.asarray(y)


def flip_batch(batch):
    "Flip the videos in the batch"

    flipped = []
    for vid in batch:
        flip_vid = [np.fliplr(frame) for frame in vid]
        flipped.append(flip_vid)

    return np.asarray(flipped)


def blur_batch(batch):
    "Blur the videos in the batch"

    blur = []
    for vid in batch:
        blur_vid = [gaussian_filter(frame, sigma=1.25) for frame in vid]
        blur.append(blur_vid)

    return np.asarray(blur)


def rotate_batch(batch):
    "Rotate the videos in the batch"

    rot_flip = []
    for vid in batch:
        angle = random.randint(0, 10)
        rot_vid = np.asarray([scipy.ndimage.interpolation.rotate(frame, angle) for frame in vid])
        rot_vid_1 = [cv2.resize(frame, (200, 95)) for frame in rot_vid]
        rot_flip.append(rot_vid_1)

    return np.asarray(rot_flip)


def add_noise(batch):
    "Add noise to the videos in the batch"

    for i in range(batch.shape[0]):
        for j in range(batch.shape[1]):
            frame_shape = batch[i, j, :, :, :].shape
            amount_of_noise = int(0.05 * frame_shape[0] * frame_shape[1])

            for k in range(amount_of_noise):
                x = np.random.randint(1, 200)
                y = np.random.randint(1, 95)

                # half salt, half pepper
                if k <= (amount_of_noise / 2):
                    batch[i, j, y, x, :] = [1, 1, 1]
                else:
                    batch[i, j, y, x, :] = [0, 0, 0]

    return np.asarray(batch)


def resize_batch(batch):
    "Resize the videos in the batch"

    random_resize = random.uniform(1, 3)
    resized = []
    new_height = 95 / random_resize
    new_weight = 200 / random_resize

    for vid in batch:
        res_vid = [cv2.resize(frame, (int(new_weight), int(new_height))) for frame in vid]
        delta_w = 200 - new_weight
        delta_h = 95 - new_height

        top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
        left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))

        color = [0, 0, 0]
        res_vid_bord = [cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) for frame
                        in res_vid]
        res_vid_bord_1 = [cv2.resize(frame, (200, 95)) for frame in res_vid_bord]
        resized.append(res_vid_bord_1)

    return np.asarray(resized)

def final_pred(predictions):
    "Create a list with the highest prediction per video"

    final_pick = []
    for prediction in predictions:
        final_pick.append(np.argmax(prediction))
    return final_pick

def generate_ytrue(paths):
    "Get labels from the test data"

    y = []
    for file in paths:
        if 'handShake' in file:
            y.append(0)
        if 'highFive' in file:
            y.append(1)
        if 'hug' in file:
            y.append(2)
        if 'kiss' in file:
            y.append(3)
    return y

def metrics(true, pred):
    "Create the values for the confusion matrix"

    true_hs = 0
    p_hs_t_hf = 0
    p_hs_t_h = 0
    p_hs_t_k = 0
    # predictions for high five
    true_hf = 0
    p_hf_t_hs = 0
    p_hf_t_h = 0
    p_hf_t_k = 0
    # predictions for hug
    true_h = 0
    p_h_t_hs = 0
    p_h_t_hf = 0
    p_h_t_k = 0
    # predictions for kiss
    true_k = 0
    p_k_t_hs = 0
    p_k_t_hf = 0
    p_k_t_h = 0

    for i, j in zip(pred, true):
        # handshake
        if i == 0 and j == 0:
            true_hs += 1
        if i == 0 and j == 1:
            p_hs_t_hf += 1
        if i == 0 and j == 2:
            p_hs_t_h += 1
        if i == 0 and j == 3:
            p_hs_t_k += 1

        # high five
        if i == 1 and j == 1:
            true_hf += 1
        if i == 1 and j == 0:
            p_hf_t_hs += 1
        if i == 1 and j == 2:
            p_hf_t_h += 1
        if i == 1 and j == 3:
            p_hf_t_k += 1

        # hug
        if i == 2 and j == 2:
            true_h += 1
        if i == 2 and j == 0:
            p_h_t_hs += 1
        if i == 2 and j == 1:
            p_h_t_hf += 1
        if i == 2 and j == 3:
            p_h_t_k += 1

        # kiss
        if i == 3 and j == 3:
            true_k += 1
        if i == 3 and j == 0:
            p_k_t_hs += 1
        if i == 3 and j == 1:
            p_k_t_hf += 1
        if i == 3 and j == 2:
            p_k_t_h += 1

    print('True Handshakes:', true_hs)
    print('p_hs_t_hf:', p_hs_t_hf)
    print('p_hs_t_h:', p_hs_t_h)
    print('p_hs_t_k:', p_hs_t_k)
    print('True Highfive:', true_hf)
    print('p_hf_t_hs:', p_hf_t_hs)
    print('p_hf_t_h:', p_hf_t_h)
    print('p_hf_t_k:', p_hf_t_k)
    print('True Hugs:', true_h)
    print('p_h_t_hs:', p_h_t_hs)
    print('p_h_t_hf:', p_h_t_hf)
    print('p_h_t_k:', p_h_t_k)
    print('True Kiss:', true_k)
    print('p_k_t_hs:', p_k_t_hs)
    print('p_k_t_hf:', p_k_t_hf)
    print('p_k_t_h:', p_k_t_h)

# creating two sets of paths. The videos have been ordered in the maps in the original split
# with the first 5 videos shifted from set 1 to set 2
paths = glob.glob('/Users/noaleenders/Thesis/code/Final_data/Set_1/**/*.mp4')
test_paths = glob.glob('/Users/noaleenders/Thesis/code/Final_data/Set_2/**/*.mp4')

# split paths in train and validation paths
train_paths, val_paths = train_test_split(paths, test_size=0.2, shuffle=True)

# create three generators
train_gen = VidDataGenerator(train_paths, batch_size=8, n_classes=4, shuffle=True, to_fit=True)
val_gen = VidDataGenerator(val_paths, batch_size=8, n_classes=4, shuffle=False, to_fit=True)
test_gen = VidDataGenerator(test_paths, batch_size=1, n_classes=4, shuffle=False, to_fit=False, aug=False)

# creating the 3D CNN
model = Sequential()
model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', input_shape=(10, 95, 200, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool3D(pool_size=(1, 2, 2), padding='valid'))
model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool3D(pool_size=(1, 2, 2), padding='valid'))
model.add(layers.Conv3D(256, (3, 3, 3), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(4, activation="softmax"))

# the use of the Adam optimizer with learning rate 0.00001
opt = keras.optimizers.Adam(lr=0.00001)

# setting early stopping
earlystop = EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True)
callbacks = [earlystop]

# compiling the model
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=["accuracy"])

# fitting the model
model.fit(train_gen, validation_data=val_gen, epochs=35, callbacks=callbacks)

# saving the model
model.save("3DCNN_model")

# creating predictions with the created model
predictions = model.predict_generator(test_gen)

# changing the 4 predictions in a list of only the highest predictions
y_pred = final_pred(predictions)

# creating a list with the true predictions
y_true = generate_ytrue(test_paths)

# creating the right numbers for the confusion matrix
conf_matrix = metrics(y_true, y_pred)