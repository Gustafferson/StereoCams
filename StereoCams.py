import cv2   #include opencv library functions in python
import numpy as np
from pathlib import Path
from tqdm import tqdm


def captureCalibration(savePath):
    """
    Inputs: 
        savePath: the path to the save location the calibration folders and images
    """
    camLeft     = cv2.VideoCapture(0)
    camRight    = cv2.VideoCapture(1)

    cv2.namedWindow("left cam")
    cv2.namedWindow("right cam")


    img_counter = 0

    # pathName = Path('D:/StereoCams/Test')
    pathName = Path(savePath)

    Path(pathName/"Calibration").mkdir(parents=True,exist_ok=True)
    Path(pathName/"Calibration"/"Left").mkdir(parents=True,exist_ok=True)
    Path(pathName/"Calibration"/"Right").mkdir(parents=True,exist_ok=True)

    leftCalibrationDir = Path(pathName/"Calibration"/"Left")
    rightCalibrationDir = Path(pathName/"Calibration"/"Right")

    while True:
        ret, frameLeft = camLeft.read()
        ret, frameRight = camRight.read()

        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("left cam", np.flipud(frameLeft))
        cv2.imshow("right cam",frameRight)

        k = cv2.waitKey(1)
        print("Press space to capture, escape to finish")
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "calibration_frame_{}.png".format(str(img_counter).zfill(2))
            cv2.imwrite(str(  (leftCalibrationDir / img_name)  ), frameLeft)
            cv2.imwrite(str(  (rightCalibrationDir / img_name)  ), frameRight)

            print("Dual {} written!".format(img_name))
            img_counter += 1

    camLeft.release()
    camRight.release()

    print('Cameras killed')

    cv2.destroyAllWindows()

def yesno(question):
    """Simple Yes/No Function."""
    prompt = f'{question} ? (y/n): '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return yesno(question)
    if ans == 'y':
        return True
    return False


def checkerboardSeriesLocator(checkerboard_path,checkerboard_size=(8,6),show_example=False):
    """
    Finds the checkerboard points in a given series of checkerboard images 
    and returns an array of points of size 2 x p number of points x n number of frames.
    Inputs:
        checkerboard_path: a path to a folder containing a series of checkerboard images
        (optional) checkerboard_size: a tuple of dimensions (biggest,smallest); default - (8,6)
        (optional) show_example: boolean; default - False; if True a figure is opened showing picked coordinates in a sample image
    Outputs:
        coordinates: a list of 2d arrays showing x, y coordinates of checkerboard corners in the image frame
        objpoints: a list of 2d arrays showing nominal x,y,z coordinates of checkerboard points in space 
        frameDims: dimensions of the image frame for this series
    """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    pathName = Path(checkerboard_path)
    globbed = pathName.glob('*.png')
    imageList = [x for x in globbed if x.is_file()]

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((checkerboard_size[0]*checkerboard_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[1],0:checkerboard_size[0]].T.reshape(-1,2)

    coordinates = np.zeros((len(imageList),np.prod(checkerboard_size),1,2),dtype=np.float32)
    objpoints = np.zeros((len(imageList),np.prod(checkerboard_size),3),dtype=np.float32) # 3d point in real world space

    pbar = tqdm(     (file for file in imageList)  ,total=len(imageList)   )
    for i, fname in enumerate(pbar):
        pbar.set_description("Locating checkerboard coordinates in {}".format(fname.name)) 

        img = cv2.imread(str(fname),0)
        ret, corners = cv2.findChessboardCorners(img,checkerboard_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

        if ret:
            corners2=cv2.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
            coordinates[i,:,:,:]=corners2[:,:,:]
            objpoints[i,:,:]=objp
        else:
            print('No checkerboard found in {}, exiting...'.format(fname.name))
            exit()
        
    frameDims = img.shape
    if show_example==True:
        cv2.namedWindow('Calibration', cv2.WINDOW_AUTOSIZE)

        ret=cv2.drawChessboardCorners(img, checkerboard_size,corners2, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if yesno('Do you want to save results'):
        print('Saving to npy...')
        np.save(str((pathName/'coordinates.npy')),coordinates)
        np.save(str((pathName/'objpoints.npy')),objpoints)

    return coordinates,objpoints,frameDims

def singleCalibration(checkerboard_path,coordinates=None,objpoints=None,frameDims=None):
    """
    Takes either prescibed or saved coordinates and objpoints and runs cv2.calibrateCamera
    inputs:
        checkerboard_path: a path to a folder containing a series of checkerboard images
        (optional) coordinates: a list of 2d arrays showing x, y coordinates of checkerboard corners in the image frame
        (optional) objpoints: a list of 2d arrays showing nominal x,y,z coordinates of checkerboard points in space 
        (optional) frameDims: dimensions of the image frame for this series
    Outputs:
        mtx: camera matrix,
        dist:  distortion coefficients,
        rvecs: rotation vectors,
        tvesc: translation vectors
    """

    pathName = Path(checkerboard_path) 
    if not coordinates:
        print('Loading coordinates from directory...')
        if (pathName/'coordinates.npy').is_file() & (pathName/'objpoints.npy').is_file():
            coordinates = np.load(str((pathName/'coordinates.npy')))
            objpoints = np.load(str((pathName/'objpoints.npy')))
        else:
            print('No coordinates given or in directory.')
            return
    else:
        coordinates=coordinates
        objpoints=objpoints
    if not frameDims:
        print('Getting frame dimensions from sample...')
        globbed = pathName.glob('*.png')
        imageList = [x for x in globbed if x.is_file()]
        img = cv2.imread(str(imageList[0]),0)
        frameDims = img.shape
    else:
        frameDims=frameDims
    
    #returns the camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, coordinates, frameDims, None, None)

    if yesno('Do you want to save results'):
            print('Saving to npy...')
            np.save(str((pathName/'mtx.npy')),mtx)
            np.save(str((pathName/'dist.npy')),dist)
            np.save(str((pathName/'rvecs.npy')),rvecs)
            np.save(str((pathName/'tvecs.npy')),tvecs)

    return mtx,dist,rvecs,tvecs


def stereoCalibration(calibration_path,objpoints=None,left_coordinates=None, right_coordinates=None,
                            left_matrix=None,right_matrix=None,left_dist=None,right_dist=None,frameDims=None):
    
    pathName = Path(calibration_path) 
    leftDir = pathName/'Left'
    rightDir = pathName/'Right'

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    if not left_coordinates:
        #load left_coordinates
        print('Loading left coordinates from directory...')
        if (leftDir/'coordinates.npy').is_file() & (leftDir/'objpoints.npy').is_file():
            left_coordinates = np.load(str((leftDir/'coordinates.npy')))
            objpoints = np.load(str((leftDir/'objpoints.npy')))
        else:
            print('Cannot find coordinates in {} directory.'.format(leftDir.name))
            return
    if not right_coordinates:
        #load right_coordinates
        print('Loading right coordinates from directory...')
        if (rightDir/'coordinates.npy').is_file() & (rightDir/'objpoints.npy').is_file():
            right_coordinates = np.load(str((rightDir/'coordinates.npy')))
            objpoints = np.load(str((rightDir/'objpoints.npy')))
        else:
            print('Cannot find coordinates in {} directory.'.format(rightDir.name))
            return
    if not left_matrix:
        #load left_matrix
        print('Loading left cam matrix from directory...')
        if (leftDir/'mtx.npy').is_file() & (leftDir/'dist.npy').is_file():
            left_matrix = np.load(str((leftDir/'mtx.npy')))
            left_dist = np.load(str((leftDir/'dist.npy')))
        else:
            print('Cannot find coordinates in {} directory.'.format(leftDir.name))
            return
    if not right_matrix:
        #load right_matrix
        print('Loading right cam matrix from directory...')
        if (rightDir/'mtx.npy').is_file() & (rightDir/'dist.npy').is_file():
            right_matrix = np.load(str((rightDir/'mtx.npy')))
            right_dist = np.load(str((rightDir/'dist.npy')))
        else:
            print('Cannot find coordinates in {} directory.'.format(rightDir.name))
            return
    if not frameDims:
        globbed = leftDir.glob('*.png')
        imageList = [x for x in globbed if x.is_file()]
        img = cv2.imread(str(imageList[0])
        frameDims = img.shape

    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints,
            left_coordinates,right_coordinates,
            left_matrix,left_dist,
            right_matrix,right_dist,frameDims
            criteria=stereocalib_criteria)
    

    camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

    cv2.destroyAllWindows()
    return camera_model


def main():
    #Run main function.
    if yesno('Do you want to capture calibration images?'):
        calibSavePath = Path(input("Give the calibration image save location"))
        captureCalibration(calibSavePath)

        leftpath = calibSavePath/"Calibration"/"Left"
        rightpath = calibSavePath/"Calibration"/"Right"

        l_coord,l_objPoints,l_frameDims = checkerboardSeriesLocator(leftpath)
        r_coord,r_objPoints,r_frameDims = checkerboardSeriesLocator(rightpath)

        l_mtx, l_dist,_,_ = singleCalibration(leftpath)
        r_mtx, r_dist,_,_ = singleCalibration(rightpath)

        cam_model = stereoCalibration((calibSavePath/'Calibration'),l_objPoints,l_coord,r_coord
                                l_mtx,r_mtx,l_dist,r_dist,l_frameDims)

        
    elif Path(Path.cwd()/'Test'/'Calibration').exists():
        #if main run from a folder containing calibration images, those will be used
        print('Using existing calibration series')
        l_coord,l_objPoints,l_frameDims = checkerboardSeriesLocator((Path.cwd()/'Test'/'Calibration'/'Left'))
        r_coord,r_objPoints,r_frameDims = checkerboardSeriesLocator((Path.cwd()/'Test'/'Calibration'/'Right'))

        l_mtx, l_dist, _,_ = singleCalibration((Path.cwd()/'Test'/'Calibration'/'Left'),l_coord,l_objPoints,l_frameDims)
        r_mtx, r_dist, _,_ = singleCalibration((Path.cwd()/'Test'/'Calibration'/'Right'),r_coord,r_objPoints,r_frameDims)
        

        cam_model = stereoCalibration((Path.cwd()/'Test'/'Calibration'),l_objPoints,l_coord,r_coord
                                            l_mtx,r_mtx,l_dist,r_dist,l_frameDims)


    else:
        print('No calibration images in expected location, exiting.')
        exit()


if __name__ == '__main__':
    main()