from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
import matplotlib.image as mpimg
from PIL import Image
print("Enter a value for selecting a Test Image")

#select image to test

test = ['subject01.centerlight.jpg','subject01.happy.jpg','subject01.normal.jpg','subject02.normal.jpg','subject03.normal.jpg','subject07.centerlight.jpg','subject07.happy.jpg','subject07.normal.jpg','subject10.normal.jpg','subject11.centerlight.jpg','subject11.happy.jpg','subject11.normal.jpg','subject12.normal.jpg','subject14.happy.jpg','subject14.normal.jpg','subject14.sad.jpg','subject15.normal.jpg','apple1_gray.jpg']
for i in range(0,18):
    print("Enter "+str(i)+" for selecting test image "+test[i])
value =int(input())
if value>17:
    print("Not a valid Number - Please select value from 0-17")
#define training images
TrainingImages = ['subject01.normal.jpg', 'subject02.normal.jpg', 'subject03.normal.jpg', 'subject07.normal.jpg', 'subject10.normal.jpg', 'subject11.normal.jpg', 'subject14.normal.jpg', 'subject15.normal.jpg']
Imageobjects = []
data = []
totalLength = len(TrainingImages)
for i in range(totalLength):
    inputValue = TrainingImages[i]
    image_object = Image.open(inputValue)
    Imageobjects.append(image_object)
    data.append(image_object)
width, height = Imageobjects[0].size
length = len(Imageobjects)
#images are stacked in the list imagesN2vector
#The mean face m(meanFace) is computed by taking the average of the M training face images
def N2andaverage(width, height):
    meanFace = np.zeros((width*height,1),dtype = np.int64)
    N2vector = []
    create=[]
    for images in range(length):
        k = 0
        l=0
        total = width*height
        eachImages = np.zeros((total,1),dtype = np.int64)
        for i in range(height):
            for j in range(width):
                l = width+height
                #storing the pixel value in the form of increasing rows with single column
                eachImages[k,0] = Imageobjects[images].getpixel((j,i))
                create.append(eachImages)
                k = k + 1
        N2vector.append(eachImages)
    val = range(len(N2vector[0]))
    #calculating mean
    for i in val:
        sum = 0
        v1 = len(N2vector)
        r = range(v1)
        for images in r:
            sum = sum + N2vector[images][i][0]
        sum = sum/len(N2vector)
        meanFace[i][0] = sum
    return meanFace,N2vector
meanFace, N2vector = N2andaverage(width,height)
print("The Mean Face is")
plt.imshow(meanFace.reshape(231,195), cmap = "gray")
plt.show()
#subtract meanface ,A ,alterate to covariance ,covarinace
def subtractFace(imagesN2, meanFace):
    subtractface = []
    cross = width*height
    facevalue = []
    r = range(length)
    for images in r:
        #subtract mean face
        individualImages = np.subtract(imagesN2[images],meanFace)
        facevalue.append(meanFace)
         #appending the individual subtracted image stacked in one list
        subtractface.append(individualImages)
    t = len(subtractface) #length of subtractafce
    A = np.zeros((cross,t),dtype = np.int16) #empty matrix 
    r1= range(cross)
    for i in r1:
        r2=range(t)
        for j in r2:
            #All training faces into a single matrix A 
            A[i,j] = subtractface[j][i][0]
            facevalue.append(A)
    #AT is the transpose of matrix A
#taking the dot product of AT and A to get the matrix L
# calculating the eigenvalue and eigen vector
# w1 is the eigenvalue, w2 is the eigenvector
    AT = np.transpose(A)
    L = np.dot(AT, A)
    w1, w2 = LA.eig(L) #eigenvalues and eigenvector
    #final is eigenspace/facespace
    final = np.dot(A, w2) 
    term = range (len(final[0]))
    for i in range (len(final[0])):
        #print eigen faces of training values
        plt.title('Eighen face of TrainingImage '+ str(i))
        img = (final[:,i].reshape(231,195))
        plt.imshow(img,cmap='gray')
        plt.show()
    return final,subtractface
final,subtractface = subtractFace(N2vector,meanFace)
def projectedFaceSpace(final, sFace):
    rows, column = np.transpose(final).shape
    projectedfacespace = []
    projected = []
    length = len(Imageobjects)
    r = range(length)
    for images in r:
        #project train space on face space
        dot = np.dot(np.transpose(final), sFace[images])
        projectedfacespace.append(dot)
        projected.append(rows)
    return projectedfacespace
projectedfacespace = projectedFaceSpace(final,subtractface )
def PCA_Coefficients(projectedfacespace):
    length = len(projectedfacespace)
    r = range(length)
    for i in r:
        #get all PCA coefficients
        print ("PCA Coefficient for training Image ",i)
        print (projectedfacespace[i])
PCA_Coefficients(projectedfacespace)
testImage = test[value]
imagevalue =  test[value]
test_image = Image.open(imagevalue)
width, height =Image.open(imagevalue).size
#reading the test image of which the face needs to be recognized
#subtracting the mean face m from each test face
def getTestandSubtractImage(width, height,meanFace):
    total = width*height
    h = range(height)
    l = range(width)
    Testimage = np.zeros((total,1),dtype = np.int64)
    k = 0
    l1 = 0
    for i in h:
        for j in l:
            Testimage[k,0] = test_image.getpixel((j,i))
            k = k + 1
            l1 = l1+1
    subtracttestface = np.subtract(Testimage,meanFace)
    return subtracttestface
I = getTestandSubtractImage(width, height,meanFace)
print("Subtracted Test Face of "+ imagevalue)
reshape = I.reshape(231,195)
plt.imshow(reshape, cmap = "gray")
plt.show()
def projectiononFace(U, I):
    #computing subtracted projection onto the face space
    trans = np.transpose(U)
    
    projectionface = np.dot(trans, I)
    #Reconstruct input face image from the eigenfaces
    reconstructedimage = np.dot(U, projectionface)
    print ("PCA coefficent of "+imagevalue+ " is")
    print(projectionface)
    return reconstructedimage,projectionface
reconstructedimage,projectionface = projectiononFace(final, I)
#printing constructed Face
print("Reconstructed Image of " + imagevalue)
shape1 = reconstructedimage.reshape(231,195)
plt.imshow(shape1, cmap = "gray")
plt.show()
#Computing the distance between the input face image and the reconstruction of the image
#use euclidean distance
def EuclideanDistance(reconstructedimage, I):
    subtractedform = LA.norm( np.subtract(reconstructedimage, I))
    print("Distance di is = "+str(subtractedform) )
    length =len(projectedfacespace)
    r = range(length)
    return subtractedform
def calculate(projection, projected):
    Di=[]
    length1 = len(projected)
    r1= range(length1)
    # distance between input face image and training images in the face space
#projection is the projected test face,projected[i] is the individual traing images
    for i in r1:
        subtract = np.subtract(projection, projected[i])
        di = LA.norm(subtract)
        Di.append(di)
    return Di

sub = EuclideanDistance(reconstructedimage, I)

#to to identify whether image is face or not 
T0 = 7345724145782
#T1 is used to identify whether the face is present in the dataset or not
T1 = 90000000
if (sub < T0):
    Di = calculate(projectionface, projectedfacespace)
    print("Distance D of the given Test Image to Train Image is " + str(min(Di)))
    if (min(Di) > T1):
        print("Unknown face")
    else:
        minimum = Di.index(min(Di))
        print("The Face with which the given face is similar to is", TrainingImages[minimum])
        plt.title('Input Test Image Given')
        plt.imshow(mpimg.imread(testImage),cmap='gray')
        plt.figure()
        minDist = Di.index(min(Di))
        plt.imshow(mpimg.imread(TrainingImages[minDist]),cmap='gray')
        plt.title('Resulting image')
        plt.show()

else:
    print ("Its Non face")
    

