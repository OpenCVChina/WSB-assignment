import os
import argparse

import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--database_dir', '-db', type=str, default='./database')
parser.add_argument('--face_detection_model', '-fd', type=str, required=True)
parser.add_argument('--face_recognition_model', '-fr', type=str, required=True)
args = parser.parse_args()

def detect_face(detector, image):
    ''' Run face detection on input image.

    Paramters:
        detector - an instance of cv.FaceDetectorYN
        image    - a single image read using cv.imread

    Returns:
        faces    - a np.array of shape [n, 15]. If n = 0, return an empty list.
    '''
    faces = []
    ### TODO: your code starts here

    ### your code ends here
    return faces

def extract_feature(recognizer, image, faces):
    ''' Run face alignment on the input image & face bounding boxes; Extract features from the aligned faces.

    Parameters:
        recognizer - an instance of cv.FaceRecognizerSF
        image      - a single image read using cv.imread
        faces      - the return of detect_face

    Returns:
        features   - a length-n list of extracted features. If n = 0, return an empty list.
    '''
    features = []
    ### TODO: your code starts here

    ### your code ends here
    return features

def match(recognizer, feature1, feature2, dis_type=1):
    ''' Calculate the distatnce/similarity of the given feature1 and feature2.

    Parameters:
        recognizer - an instance of cv.FaceRecognizerSF. Call recognizer.match to calculate distance/similarity
        feature1   - extracted feature from identity 1
        feature2   - extracted feature from identity 2
        dis_type   - 0: cosine similarity; 1: l2 distance; others invalid

    Returns:
        isMatched  - True if feature1 and feature2 are the same identity; False if different
    '''
    l2_threshold = 1.128
    cosine_threshold = 0.363
    isMatched = False
    ### TODO: your code starts here

    ### your code ends here
    return isMatched

def load_database(database_path, detector, recognizer):
    ''' Load database from the given database_path into a dictionary. It tries to load extracted features first, and call detect_face & extract_feature to get features from images (*.jpg, *.png).

    Parameters:
        database_path - path to the database directory
        detector      - an instance of cv.FaceDetectorYN
        recognizer    - an instance of cv.FaceRecognizerSF

    Returns:
        db_features   - a dictionary with filenames as key and features as values. Keys are used as identity.
    '''
    db_features = dict()

    print('Loading database ...')
    # load pre-extracted features first
    for filename in os.listdir(database_path):
        if filename.endswith('.npy'):
            identity = filename[:-4]
            if identity not in db_features:
                db_features[identity] = np.load(os.path.join(database_path, filename))
    npy_cnt = len(db_features)
    # load images and extract features
    for filename in os.listdir(database_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            identity = filename[:-4]
            if identity not in db_features:
                image = cv.imread(os.path.join(database_path, filename))
                faces = detect_face(detector, image)
                features = extract_feature(recognizer, image, faces)
                if len(features) > 0:
                    db_features[identity] = features[0]
                    np.save(os.path.join(database_path, '{}.npy'.format(identity)), features[0])
    cnt = len(db_features)
    print('Database: {} loaded in total, {} loaded from .npy, {} loaded from images.'.format(cnt, npy_cnt, cnt-npy_cnt))
    return db_features

def visualize(image, faces, identities, fps, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    output = image.copy()

    # put fps in top-left corner
    cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    for face, identity in zip(faces, identities):
        # draw bounding box
        bbox = face[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        # put identity
        cv.putText(output, '{}'.format(identity), (bbox[0], bbox[1]-15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    return output

if __name__ == '__main__':
    # Initialize FaceDetectorYN
    detector = cv.FaceDetectorYN.create(''' parameters for initialization ''')
    # Initialize FaceRecognizerSF
    recognizer = cv.FaceRecognizerSF.create(''' parameters for initialization ''')

    # Load database
    database = load_database(args.database_dir, detector, recognizer)

    # Initialize video stream
    device_id = 0
    cap = cv.VideoCapture(device_id)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Real-time face recognition
    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        tm.start()
        # detect faces
        faces = detect_face(detector, frame)
        # extract features
        features = extract_feature(recognizer, frame, faces)
        # match detected faces with database
        identities = []
        for feature in features:
            isMatched = False
            for identity, db_feature in database.items():
                isMatched = match(recognizer, feature, db_feature)
                if isMatched:
                    identities.append(identity)
                    break
            if not isMatched:
                identities.append('Unknown')
        tm.stop()

        # Draw results on the input image
        frame = visualize(frame, faces, identities, tm.getFPS())

        # Visualize results in a new Window
        cv.imshow('Face recognition system', frame)

        tm.reset()