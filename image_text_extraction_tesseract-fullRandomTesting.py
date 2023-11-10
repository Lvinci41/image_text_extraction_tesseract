import cv2 as cv
import os
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
import random as rand 

def coinFlip(x):
    return False if rand.random() < x else True

#tesseract file pointer
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#set up
current_path = os.listdir(os.getcwd())
border = "********************"
csv_cols = ['line', 'Case Number', 'Case Name', 'Image', 'Thresholding Used', 'Blur1', 'Blur2', 'Canny', 'deNoise', 'CannyMin', 'CannyMax', 'binaryThreshold','BlockSize', 'C', 'Blur1Median', 'Blur2Median', 'deNoiseH', 'deNoiseTemplate', 'deNoiseSearch', 'errorScore']
tests_summary_df = pd.DataFrame(columns=csv_cols)

"""
**************************************************************************
Get word counts from "ref" files (from wordCountTest_testingRandom_1.py)
**************************************************************************
"""
refTranscriptionFiles = [refFile[:len(str(refFile))-4:] for refFile in current_path if refFile[len(str(refFile))-7:] == "ref.txt"]
evalTextFiles = [evalFile[:len(str(evalFile))-4:] for evalFile in current_path if evalFile[len(str(evalFile))-7:] == "val.txt"]
transcriptionsDict_word = dict(zip(refTranscriptionFiles, [None]*len(refTranscriptionFiles)))
transcriptionsDict_line = dict(zip(refTranscriptionFiles, [None]*len(refTranscriptionFiles)))

d = dict() 
lineList = []
for refFile in refTranscriptionFiles:
    text = open(refFile+".txt", "r")
    d = dict() 
    lineList = []
    lineList = [line.strip() for line in text]
    for line in lineList: 
        line_s = line.strip() 
        words = line_s.split(" ")
        for word in words: 
            if word == '':
                continue
            if word in d: 
                d[word] += 1
            else: 
                d[word] = 1
    transcriptionsDict_word[refFile] = d
    transcriptionsDict_line[refFile] = lineList

for file in list(transcriptionsDict_word.keys()):
    running_sum = 0
    for word in list(transcriptionsDict_word[file].keys()):
        running_sum += transcriptionsDict_word[file][word]
    transcriptionsDict_word[file]["$$WORD_TOTAL$$"] = running_sum
"""
**************************************************************************
/END Get word counts from "ref" files (from wordCountTest_testingRandom_1.py)
**************************************************************************
"""

test_no=1
line_no=1
while test_no < 9:
    thresh_case = rand.randint(0,4)
    if thresh_case == 0:
        canny_case = coinFlip(0.4)
        blur1_case = coinFlip(0.4)
        blur2_case = coinFlip(0.4)
    else:
        canny_case = coinFlip(0.8)
        blur1_case = coinFlip(0.8)
        blur2_case = coinFlip(0.8)        
    thresh_case_name = ["simpleGray", "binary", "adaptiveMean", "adaptiveGauss", "otsu"]
    dnoise_case = rand.randint(1,3)# coinFlip(0.2)
    dnoise_case_name = ["noDenoise", "color", "grayBefore", "grayAfter"]

    canny_max=rand.randint(10,900)
    canny_min=int( canny_max*rand.randint(1,9)/10 )
    binary_thresh=int( 255*rand.randint(1,9)/10 ) #for binary and otsu
    adaptive_blockSize=rand.randrange(3,51,2) #apparently this must be odd 
    adaptive_c=max( int(adaptive_blockSize*rand.randint(0,5)/10) ,1)
    blur1_median=rand.randrange(3,11,2) #must be odd and >1
    blur2_median=rand.randrange(3,11,2)
    dnoise_h=rand.randrange(3,11,2)
    dnoise_search=rand.randrange(3,51,2)
    dnoise_template=max( int(dnoise_search*rand.randint(0,5)/10),1)
    

    case_description="case"+str(test_no)

    if canny_case:
        case_description=case_description+"_canny="+str(canny_max)+"-"+str(canny_min)
    if blur1_case:
        case_description=case_description+"_blur1="+str(blur1_median)
    case_description=case_description+"_"+thresh_case_name[thresh_case]
    if thresh_case==1 or thresh_case==4:
        case_description=case_description+"-median="+str(binary_thresh)
    if thresh_case==2 or thresh_case==3:
        case_description=case_description+"-block="+str(adaptive_blockSize)+"-c="+str(adaptive_c)
    if blur2_case:
        case_description=case_description+"_blur2="+str(blur2_median)
    if dnoise_case==0:
        case_description=case_description+"_noDenoise"
    else:
        case_description=case_description+"_"+dnoise_case_name[dnoise_case]+"="+str(dnoise_h)+"-"+str(dnoise_template)+"-"+str(dnoise_search)

    for image in current_path:
        #check if the file is a valid image
        img = cv.imread(image)
        if img is None: continue
        if image[-7:-4] != "val": continue  ##only for evaluations where testing files end in "val"

        if dnoise_case==1:
            img=cv.fastNlMeansDenoisingColored(img, None, dnoise_h, dnoise_h, dnoise_template, dnoise_search )


        gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if dnoise_case==2:
            gray=cv.fastNlMeansDenoising(gray, None, dnoise_h, dnoise_template, dnoise_search )        
        if blur1_case:
            gray=cv.medianBlur(gray, blur1_median)   
        if thresh_case==1:
            ret, gray = cv.threshold(gray,binary_thresh,255,cv.THRESH_BINARY)
        if thresh_case==2:
            gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,adaptive_blockSize,adaptive_c)       
        if thresh_case==3:
            gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,adaptive_blockSize,adaptive_c)               
        if thresh_case==4:
            ret, gray = cv.threshold(gray, binary_thresh, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)       
        if canny_case:
            gray=cv.Canny(gray, canny_min, canny_max)
        if blur2_case:
            gray=cv.medianBlur(gray, blur2_median) 
        if dnoise_case==3:
            gray=np.uint8(gray)
            gray= cv.fastNlMeansDenoising(gray, None, dnoise_h, dnoise_template, dnoise_search ) #could also be NORM_L1, find out the difference 
    
        filename = "{}".format(str(case_description)+"_"+str(image))
        cv.imwrite(filename, gray)
        text = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)

        with open(str(filename)[:-4]+"_"+str(image)[-3:]+"_result.txt", "a+") as file:
            file.write(text)

        file.close()
        readFile = open(str(filename)[:-4]+"_"+str(image)[-3:]+"_result.txt", "r")        

        d1 = dict()
        lineList1 = []
        lineList1 = [line.strip() for line in readFile]
        for line in lineList1: 
            words = line.split(" ")
            for word in words: 
                if word == '':
                    continue
                if word in d1: 
                    d1[word] += 1
                else: 
                    d1[word] = 1
        
        readFile.close()

        refWordList = list( transcriptionsDict_word[image[:len(str(image))-7]+"ref"].keys() )

        absolute_error = 0
        for word in refWordList:
            if word == "$$WORD_TOTAL$$": continue
            if word in list(d1.keys()):
                absolute_error += abs( transcriptionsDict_word[image[:len(str(image))-7]+"ref"][word] - d1[word] )
            else:
                absolute_error += transcriptionsDict_word[image[:len(str(image))-7]+"ref"][word]

        new_row = [line_no, test_no, filename, image, thresh_case_name[thresh_case], blur1_case, blur2_case, canny_case, dnoise_case_name[dnoise_case], canny_max, canny_min, binary_thresh, adaptive_blockSize, adaptive_c, blur1_median, blur2_median, dnoise_h, dnoise_template, dnoise_search, round( absolute_error/transcriptionsDict_word[image[:len(str(image))-7]+"ref"]["$$WORD_TOTAL$$"], 5) ]
        tests_summary_df=tests_summary_df._append(pd.Series(new_row, index=tests_summary_df.columns, name=str(line_no)), ignore_index=True)

        line_no+=1
    test_no +=1

tests_summary_df.to_csv('test_summary.csv', index=False)