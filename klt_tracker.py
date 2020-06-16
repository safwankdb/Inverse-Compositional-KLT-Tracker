import numpy as np
import cv2
from petyr import Affine, Similarity

bbox = -np.ones((4, 2), np.int32)
template = np.zeros((480, 640), np.int)


def warp(p):
    p = np.array(p)
    p = p.ravel()
    assert p.shape[0] == 2
    return Similarity().translate(p[0], p[1])


def KLT(img, template, a, b, c, d):
    global bbox, tbox
    img_y, img_x = np.gradient(template.astype(np.float))
    img_x = img_x[a:b, c:d].ravel()
    img_y = img_y[a:b, c:d].ravel()

    template = template[a:b, c:d]

    n = img_x.shape[0]

    del_I = np.dstack([img_x, img_y]).reshape(-1, 1, 2)
    dW_dp = np.zeros((n, 2, 2))
    dW_dp[:, 0, 0] = 1
    dW_dp[:, 1, 1] = 1
    sd = del_I @ dW_dp
    H = (sd.transpose(0, 2, 1) @ sd).sum(0)
    p = [0, 0]

    p = np.array(p)
    eps = 0.03
    for itr in range(20):

        M = warp(p).invert().numpy()[:2, :]
        h, w = img.shape[1], img.shape[0]
        img_w = cv2.warpAffine(img.astype(np.float), M,
                               (h, w)).astype(np.int)[a:b, c:d]
        error = (img_w - template).ravel()
        A = (sd.transpose(0, 2, 1) *
             error[:, np.newaxis, np.newaxis]).sum(axis=0)
        del_p = (np.linalg.solve(H, A)).ravel()
        p = p - del_p
        err = np.sqrt(np.square(del_p).sum())
        if(err < eps):
            break
    return warp(p)


def boundary_check(x, h=480):
    x = max(min(x, h-1), 0)
    return x


def bounding_rectangle(points, pad_x=0, pad_y=0):
    xmin = np.min(points[:,0]) - pad_x
    xmax = np.max(points[:,0]) + pad_x
    ymin = np.min(points[:,1]) - pad_y
    ymax = np.max(points[:,1]) + pad_y
    new_points = np.array([[xmin, ymin], [xmax, ymin], [
        xmax, ymax], [xmin, ymax]], np.int)
    return new_points


def apply_tracker(keypoints, img, template, bbox):
    bc = boundary_check
    assert img.shape == template.shape, "{}, {}".format(
        img.shape, template.shape)
    new_points = np.zeros_like(keypoints)
    h, w = img.shape
    x0, y0 = bbox[0, :]

    img = cv2.GaussianBlur(img.astype(np.float), (5,5), 0.2)

    for i, k in enumerate(keypoints):
        x, y = k
        a, b, c, d = bc(y-12, h), bc(y+13, h), bc(x-12, w), bc(x+13, w)
        if (b-a)*(d-c) == 0:
            continue
        transform = KLT(img, template, a, b, c, d)
        x_, y_ = np.round(
            (transform * np.array([[x-x0, y-y0]]))[0], 0)
        new_points[i, :] = x_+x0, y_+y0
    points, new_bbox = apply_transform(new_points, keypoints, img, bbox)
    return points, new_bbox


def get_keypoints(img, bbox):
    (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox.astype(int))
    sub_img = img[ymin:ymin+boxh, xmin:xmin+boxw].astype(np.uint8)
    keypoints = cv2.goodFeaturesToTrack(sub_img, 20, 0.05, 2)
    if keypoints is None:
        return keypoints
    keypoints = keypoints.reshape(-1, 2) + bbox[0, :]
    keypoints = keypoints.astype(np.int)
    return keypoints


def apply_transform(new_points, keypoints, img, bbox):
    THRESHOLD = 1
    at = Similarity.from_points(keypoints, new_points)
    at = Similarity(at.numpy().round(5))

    pr_keypoints = at * keypoints
    err = np.square(new_points - pr_keypoints).sum(axis=1)
    kpt_inliers = keypoints[err < THRESHOLD]
    new_inliers = new_points[err < THRESHOLD]

    at = Similarity.from_points(kpt_inliers, new_inliers)
    w = bbox[2,0] - bbox[0,0]
    h = bbox[2,1] - bbox[0,1]
    bbox = bounding_rectangle(keypoints, w / 10, h / 10)
    bbox = at * bbox
    bbox = bounding_rectangle(bbox)
    return new_inliers, bbox


# cap = cv2.VideoCapture('/dev/video2')
frame = np.zeros((480, 640, 3), np.uint8)
drawing = False

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Easy.mp4')
img = frame.copy()
counter = 0
keypoints = None
counter = 0


while True:
    img_old = img.copy()

    # frame = cv2.imread('test.jpg')
    _, frame = cap.read()
    # for _ in range(6):
    #     _, _ = cap.read()

    counter = min(counter+1, 10)
    if counter < 2:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int)
        continue

    # if counter == 2:
    #     frame = cv2.imread('test.jpg')
    #     img_old = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int)
    #     keypoints = None
    #     org = frame.copy()
    # else:
    #     center = np.array([frame.shape[1], frame.shape[0]])/2
    #     tx, ty = np.random.randint(-4, 5, 2)
    #     at = Affine().translate(tx, ty)
    #     org = cv2.warpAffine(
    #         org, at._M[:2, :], (frame.shape[1], frame.shape[0]))
    #     frame = org.copy()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int)

    if not drawing:
        r = cv2.selectROI('img', frame, False, False)
        xmin, ymin, boxw, boxh = r
        xmax = xmin + boxw
        ymax = ymin + boxh
        bbox = np.array([[xmin, ymin], [xmax, ymin], [
                        xmax, ymax], [xmin, ymax]], np.int)
        keypoints = get_keypoints(img, bbox)
        drawing = True
        print("Press R to reselect anytime, Q to quit.")
        continue

    if keypoints is None:
        keypoints = get_keypoints(img, bbox)
    keypoints, bbox = apply_tracker(
        keypoints, img.copy(), img_old.copy(), bbox)

    if len(keypoints) < 18:
        keypoints = get_keypoints(img, bbox)

    cv2.polylines(
        frame, [bbox.reshape(-1, 1, 2).astype(np.int)], True, (0, 255, 0), 1)
    for i in keypoints:
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 3, 255, 2)
    cv2.imshow('img', frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('r'):
        bbox = -np.ones((4, 2), np.int)
        drawing = False
        keypoints = None

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
