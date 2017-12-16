from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
from PIL import Image
print("Enter a value for selecting a Test Image")
print("Enter 1 for subject01.centerlight.jpg")
print("Enter 2 for subject01.happy.jpg")
print("Enter 3 for subject01.normal.jpg")
print("Enter 4 for subject02.normal.jpg")
print("Enter 5 for subject03.normal.jpg")
print("Enter 6 for subject07.centerlight.jpg")
print("Enter 7 for subject07.happy.jpg")
print("Enter 8 for subject07.normal.jpg")
print("Enter 9 for subject10.normal.jpg")
print("Enter 10 for subject11.centerlight.jpg")
print("Enter 11 for subject11.happy.jpg")
print("Enter 12 for subject11.normal.jpg")
print("Enter 13 for subject12.normal.jpg")
print("Enter 14 for subject14.happy.jpg")
print("Enter 15 for subject14.normal.jpg")
print("Enter 16 for subject14.sad.jpg")
print("Enter 17 for subject15.normal.jpg")
print("Enter 18 for apple1_gray.jpg")
print("Enter Value:")
value =input()
def select(value):
    image = ""
    if value =="1":
        image ="subject01.centerlight.jpg"
    if value == "2":
        image ="subject01.happy.jpg"
    if value == "3":
        image ="subject01.normal.jpg"
    if value == "4":
        image ="subject02.normal.jpg"
    if value == "5":
        image ="subject03.normal.jpg"
    if value == "6":
        image ="subject07.centerlight.jpg"
    if value == "7":
        image ="subject07.happy.jpg"
    if value == "8":
        image ="subject07.normal.jpg"
    if value == "9":
        image ="subject10.normal.jpg"
    if value == "10":
        image ="subject11.centerlight.jpg"
    if value == "11":
        image ="subject11.happy.jpg"
    if value == "12":
        image ="subject11.normal.jpg"
    if value == "13":
        image ="subject12.normal.jpg"
    if value == "14":
        image ="subject14.happy.jpg"
    if value == "15":
        image ="subject14.normal.jpg"
    if value == "16":
        image ="subject14.sad.jpg"
    if value == "17":
        image ="subject15.normal.jpg"
    if value == "18":
        image ="apple1_gray.jpg"
    return image
imagevalue = select(value)
TrainingImages = ['subject01.normal.jpg', 'subject02.normal.jpg', 'subject03.normal.jpg', 'subject07.normal.jpg', 'subject10.normal.jpg', 'subject11.normal.jpg', 'subject14.normal.jpg', 'subject15.normal.jpg']
Imageobjects = []
data = []
totalLength = len(TrainingImages)
for i in range(totalLength):
    image_object = Image.open(TrainingImages[i])
    Imageobjects.append(image_object)
    data.append(image_object)
width, height = Imageobjects[0].size
length = len(Imageobjects)
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
                eachImages[k,0] = Imageobjects[images].getpixel((j,i))
                create.append(eachImages)
                k = k + 1
        N2vector.append(eachImages)
    val = range(len(N2vector[0]))
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
        individualImages = np.subtract(imagesN2[images],meanFace)
        facevalue.append(meanFace)
        subtractface.append(individualImages)
    t = len(subtractface)
    A = np.zeros((cross,t),dtype = np.int16)
    r1= range(cross)
    for i in r1:
        r2=range(t)
        for j in r2:
            A[i,j] = subtractface[j][i][0]
            facevalue.append(A)
    AT = np.transpose(A)
    L = np.dot(AT, A)
    w1, w2 = LA.eig(L) #eigenvalues
    final = np.dot(A, w2)
    term = range (len(final[0]))
    for i in term:
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
        dot = np.dot(np.transpose(final), sFace[images])
        projectedfacespace.append(dot)
        projected.append(rows)
    return projectedfacespace
projectedfacespace = projectedFaceSpace(final,subtractface )
def PCA_Coefficients(projectedfacespace):
    length = len(projectedfacespace)
    r = range(length)
    for i in r:
        print ("PCA Coefficient for training Image ",i)
        print (projectedfacespace[i])
PCA_Coefficients(projectedfacespace)
test_image = Image.open(imagevalue)
width, height =Image.open(imagevalue).size
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
    trans = np.transpose(U)
    projectionface = np.dot(trans, I)
    reconstructedimage = np.dot(U, projectionface)
    print ("PCA coefficent of "+imagevalue+ " is")
    print(projectionface)
    return reconstructedimage,projectionface
reconstructedimage,projectionface = projectiononFace(final, I)
print("Reconstructed Image of " + imagevalue)
shape1 = reconstructedimage.reshape(231,195)
plt.imshow(shape1, cmap = "gray")
plt.show()
def EuclideanDistance(reconstructedimage, I):
    Di=[]
    subtractedform = LA.norm( np.subtract(reconstructedimage, I))
    print("Distance di is = "+str(subtractedform) )
    length =len(projectedfacespace)
    r = range(length)
    for i in r:
        subtract = np.subtract(projectionface, projectedfacespace[i])
        di = LA.norm(subtract)
        Di.append(di)
    return Di
Di = EuclideanDistance(reconstructedimage, I)
print("Given Test Image is Matched with = " + TrainingImages[Di.index(min(Di))])
plt.show()
