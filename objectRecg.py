## Thomas nevin K
## Yitzak Hernandez
## UCF
'''
images must be saved on a folder called "images" located on same place as project.py

This program uses template matching, histogram matching and sift matching to see if an image is similar to a group of images.
Images are scored between 1 and however many are in the images folder. with 1 being the highest.
Top four images are in no particular order.
----->This program is the updated one for python3 and above<----

Lines 56, 57, 60, 61, 64, 65 are commented out since all results are printed on screen. Just uncomment to see results.
'''

import cv2
import numpy as np
import os

def main():
    #Takes all of the images on the file specified folder
    mypath = "images/"
    fileNames = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]
    #create a list all files in directory and
    #append files with a vaild extention to fileNames
    for file in os.listdir(mypath):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        fileNames.append(os.path.join(mypath, file))   
        
    n=0
    imd = np.empty(len(fileNames), dtype=object)
    #loop through image_path_list to open each image
    for imagePath in fileNames:
        image = cv2.imread(imagePath)
        
        # display the image on screen with imshow()
        # after checking that it loaded
        if image is not None:
            cv2.imshow(imagePath, image)
            imd[n] = cv2.imread(os.path.join(mypath,fileNames[n]))
            n=n+1
        elif image is None:
            print ("Error loading: " + imagePath)
            #end this loop iteration and move on to next image
            continue

    #Goes through each and every single image setting the first one as query, and all as database.
    for n in range(len(imd)):
        print (n)
        imq = imd[n].copy()
        
        templateTopFour, templateMatchingScores = templateMatching(imd, imq, n)
        #topFourImages(templateTopFour, fileNames, n, "Template Matching")
        #imageScorePlacing(templateMatchingScores, fileNames, n, "Template Matching")
        
        histogramTopFour, histogramMatchingScores = histogramMatching(imd, imq, n)
        #topFourImages(histogramTopFour, fileNames, n, "Histogram Matching")
        #imageScorePlacing(histogramMatchingScores, fileNames, n, "Histogram Matching")
                                    
        siftTopFour, siftMatchingScores = SIFT(imd, imq, n)
        #topFourImages(siftTopFour, fileNames, n, "SIFT")
        #imageScorePlacing(siftMatchingScores, fileNames, n, "SIFT")
        
        print
    print("Done.")



def templateMatching(imd, imq, imageNumber):
    methods = ['cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED'] #[3, 1]
    
    #dict to hold template matching results.
    topFour = {}
    totalScores = {}
    
    for n in range(0, len(imd)):
        im = imd[n].copy()
        resArr = []
        
        for meth in methods:
            method = eval(meth)
            
            # Apply template Matching
            res = cv2.matchTemplate(im,imq,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            #Save each template score into an array. We take the inverse on TM_SQDIFF_NORMED
            if eval(meth) == 1:
                res = 1 - res
                resArr.append(res)
            else:
                resArr.append(res)
        pass
        #add each score of the array and take average
        score = np.sum(resArr) / len(resArr)
        totalScores[n] = score
        
        #save top 4 scores
        if len(topFour) < 4:
            topFour[n] = score
        else:
            topFour = topFourDictionary(topFour, n, score)
    
    return topFour, totalScores


def histogramMatching(imd, imq, imageNumber):
    #dict to hold final histogram matching results.
    topFour = {}
    totalScores = {}
    
    #calculate histogram of query image.
    histq = cv2.calcHist([imq], [0], None, [256], [0, 256])
    
    #calculate histogram of each database image.
    histd = np.empty(len(imd), dtype=object)
    for n in range(0, len(imd)):
        histd[n] = cv2.calcHist([imd[n]], [0], None, [256], [0, 256])
        
    #go through all the methods on each image and take average.
    for n in range(0, len(imd)):
        resArr = []
        resArr.append(cv2.compareHist(histq, histd[n], cv2.HISTCMP_CORREL))
        resArr.append(1 - cv2.compareHist(histq, histd[n], cv2.HISTCMP_BHATTACHARYYA))
        
        score = np.sum(resArr) / len(resArr)
        totalScores[n] = score
        
        #save top 4 scores
        if len(topFour) < 4:
            topFour[n] = score
        else:
            topFour = topFourDictionary(topFour, n, score)
            
    return topFour, totalScores



def SIFT(imd, imq, imageNumber):
    #dict to hold final SIFT matching results.
    topFour = {}
    totalScores = {}
    
    for z in range(0, len(imd)):
        img1 = imq.copy()
        img2 = imd[z].copy()
        
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
                
        score = len(good)
        totalScores[z] = score
        
        #save top 4 scores
        if len(topFour) < 4:
            topFour[z] = score
        else:
            topFour = topFourDictionary(topFour, z, score)
            
    return topFour, totalScores

#We save the location in the fileNames array (n), which holds the image name, and its score (score).
#If the new image score is lower we swap.
def topFourDictionary(topFour, n, score):
    for j, k in topFour.items():
        if k < score:
            del topFour[j]
            topFour[n] = score
            return topFour
        
    return topFour

def topFourImages(topFour, fileNames, image, method):
    print(fileNames[image] + " top matches in: " + method)
    
    for n, m in topFour.items():
        print (fileNames[n])
    print

def imageScorePlacing(allScores, fileNames, image, method):
    print(fileNames[image] + " all scores in: " + method)
    count = len(allScores)
    #iteritems() is for python2 ,use items() for python3 && for lambda (k, v): (v, k) ,use lambda kv: (kv[1], kv[0]) for python3
    for key, value in sorted(allScores.items(), key = lambda kv: (kv[1], kv[0])):
        print (count, fileNames[key])
        count = count - 1
        
    print



if __name__ == "__main__":
    main()
