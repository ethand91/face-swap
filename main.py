import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils
import argparse

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (1 - mask) + img2_rect * mask

def face_swap(image1_path, image2_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    rects1 = detector(gray1, 1)
    rects2 = detector(gray2, 1)

    if len(rects1) == 0 or len(rects2) == 0:
        print("Error: Could not detect faces in one or both images.")
        return

    shape1 = predictor(gray1, rects1[0])
    shape2 = predictor(gray2, rects2[0])

    points1 = face_utils.shape_to_np(shape1)
    points2 = face_utils.shape_to_np(shape2)

    hullIndex = cv2.convexHull(points2, returnPoints=False)
    hull1 = points1[hullIndex[:, 0]]
    hull2 = points2[hullIndex[:, 0]]

    rect = (0, 0, gray2.shape[1], gray2.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(hull2.tolist())
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        indices = []
        for pt in pts:
            ind = np.where((hull2 == pt).all(axis=1))
            if len(ind[0]) == 0:
                continue
            indices.append(ind[0][0])
        indexes_triangles.append(indices)

    img2_new_face = np.zeros_like(image2)

    for indices in indexes_triangles:
        t1 = [hull1[indices[0]], hull1[indices[1]], hull1[indices[2]]]
        t2 = [hull2[indices[0]], hull2[indices[1]], hull2[indices[2]]]

        warp_triangle(image1, img2_new_face, t1, t2)

    mask = np.zeros_like(gray2)
    cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))
    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
    output = cv2.seamlessClone(img2_new_face, image2, mask, center, cv2.NORMAL_CLONE)

    cv2.imwrite("output.jpg", output)

def main():
    parser = argparse.ArgumentParser(description="Face Swapping Application")
    parser.add_argument("image1", type=str, help="Path to the first image (source face)")
    parser.add_argument("image2", type=str, help="Path to the second image (destination face)")

    args = parser.parse_args()

    face_swap(args.image1, args.image2)

if __name__ == "__main__":
    main()
