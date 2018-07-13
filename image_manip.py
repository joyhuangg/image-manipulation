#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:57:47 2017

@author: JoyHuang
"""
from sympy.plotting import plot
import numpy as np
import matplotlib.pyplot as plt
from math import *
import random

#creates sample image
def createim():
    imtest=np.array([[[0.0,0.0,0.0] for _ in range(100)] for _ in range(100)])
    for i in range(int(100/3)):
        for k in range(100):
            imtest[i][k][0] = 1.0
            imtest[i][k][1] = 1.0
            imtest[i][k][2] = 1.0
    for k in range(int(100/3), int(200/3)):
        for i in range(100):
            imtest[i][k][0] = 1/2
            imtest[i][k][1] = 1/2
            imtest[i][k][2] = 1/2
    #make a diagonal
    for i in range(100):
        j = i
        imtest[j][i][0] = 1.0
        imtest[j][i][1] = 1.0
        imtest[j][i][2] = 1.0
    
    return imtest

#adds noise to the image
def saltnpepper(image, pw, pb):
    image = np.array(image)
    row = image.shape[0]
    col = image.shape[1]
    dim = row * col
    S = pw * dim
    P = pb * dim
    Sin = 0
    Pin = 0
    usedPix = []
    while Sin < S or Pin < P:
        r = random.gauss(0, 0.2)
        i = random.randint(0, row-1)
        k = random.randint(0,col-1)
        #if random number r is greater than 0 we'll add white noise
        if r > 0 and Sin  < S and [i,k] not in usedPix:
            image[i][k][0] = 0
            image[i][k][1] = 0
            image[i][k][2] = 0
            usedPix.append([i,k])
            Sin += 1
        #if random number r is less than or = 0 we'll add black noise
        if r <= 0 and Pin < P and [i,k] not in usedPix:
            image[i][k][0] = 1
            image[i][k][1] = 1
            image[i][k][2] = 1
            usedPix.append([i,k])
            Pin += 1
    return image
            
    
###takes a gray-scale picture, and offers two options for noise removal:
###uniform or gaussian.
###input:
###@image: a gray-scale picture
###@option: either 'u' for uniform or 'g' for gaussian
###@*param: input either a list with parameter k, an odd int in for uniform 
###or a list with parameter k and sigma in for gaussian, must be in this order
###@output:
###the image with noise removal
def blurring(image, option = 'g', *param):
    image = np.array(image)
    row = image.shape[0]
    col = image.shape[1]
    update = np.array([[[0.0,0.0,0.0] for _ in range(row)] for _ in range(col)])
    k = param[0]
    if len(param) == 2:
        sig = param[1]
    #uniform noise removal
    if option == 'u':
        for i in range(row):
            for j in range(col):
                #make sure not to include edges
                if i >= floor(k/2) and i <= row-ceil(k/2) and j >= floor(k/2) and j <= col - ceil(k/2):
                    sum = 0
                    #average the pixels around i,j of k by k square
                    for m in range(i-floor(k/2), i+ceil(k/2)):
                        for n in range(j-floor(k/2), j+ceil(k/2)):
                            sum += image[m][n][0]/k**2
                    update[i][j][0] = sum
                    update[i][j][1] = sum
                    update[i][j][2] = sum
                else:
                    update[i][j][0] = image[i][j][0]
                    update[i][j][1] = image[i][j][1]
                    update[i][j][2] = image[i][j][2]
    
    #gaussian noise removal
    if option == 'g':
        update2 = np.array([[[0.0,0.0,0.0] for _ in range(row)] for _ in range(col)])
        for i in range(row):
            for j in range(col):
                #make sure not to include edges
                if i >= floor(k/2) and i <= row-ceil(k/2) and j >= floor(k/2) and j <= col-ceil(k/2):
                    sum = 0
                    #create a matrix of weights
                    for m in range(i-floor(k/2), i+ceil(k/2)):
                        for n in range(j-floor(k/2), j+ceil(k/2)):
                            g = (1/sqrt(2*pi*sig**2))*e**(-(((i-m)**2 + (j-n)**2))/(2*sig**2))
                            update2[m][n][0] = g
                            sum += g
                    #normalize the weights
                    for m in range(i-floor(k/2), i+ceil(k/2)):
                        for n in range(j-floor(k/2), j+ceil(k/2)):
                            update2[m][n][0] = update2[m][n][0]/sum
                    pix = 0
                    #average the pixels around i,j of k by k square
                    for m in range(i-floor(k/2), i+ceil(k/2)):
                        for n in range(j-floor(k/2), j+ceil(k/2)):
                            pix += update2[m][n][0]*image[m][n][0]
                    update[i][j][0] = pix
                    update[i][j][1] = pix
                    update[i][j][2] = pix
                    for m in range(i-floor(k/2), i+ceil(k/2)):
                        for n in range(j-floor(k/2), j+ceil(k/2)):
                            update2[m][n][0] = 0
                            update2[m][n][1] = 0
                            update2[m][n][2] = 0
                    
                else:
                    update[i][j][0] = image[i][j][0]
                    update[i][j][1] = image[i][j][1]
                    update[i][j][2] = image[i][j][2]                       
    return update

    
    
###takes a gray-scale image and detects edges, with the option of
###horizontal, vertical or both.
###input:
###@image: a gray-scale picture
###@option: input 'h' for horizontal, 'v' for vertical, or 'b' for both
###@output:
###the image with detected edges
def detect_edge(image, option):
    image = np.array(image)
    row = image.shape[0]
    col = image.shape[1]

    if option == 'h':
        Filter = np.array([[[0.0,0.0,0.0] for _ in range(row)] for _ in range(col)])
        F = np.array([[1,2,1], [0,0,0],[-1,-2,-1]])
        #multiply kxk box by filter F and find value of pixel i, j (pix)
        for i in range(row):
            for j in range(col):
                #make sure not to include edges
                if i >= floor(3/2) and i <= row-ceil(3/2) and j >= floor(3/2) and j <= col - ceil(3/2):
                    pix = 0
                    r = -1
                    for m in range(i-floor(3/2), i+ceil(3/2)):
                        r += 1
                        c = -1
                        for n in range(j-floor(3/2), j+ceil(3/2)):
                            c+= 1
                            pix += F[r][c]*image[m][n][0]
                    Filter[i][j][0]= pix
                    Filter[i][j][1]= pix
                    Filter[i][j][2]= pix
                    
                    
        #find max difference betwen pixel differences
        maxdiff = 0
        for i in range(col):
            for j in range(row-1):
                if abs(Filter[j][i][0] - Filter[j+1][i][0] > maxdiff):
                    maxdiff = abs(Filter[j][i][0] - Filter[j+1][i][0])
        #set an arbitrary boundary of what the amount of the minimum difference 
        #between 2 pixels that constitute an edge is
        mindiff = maxdiff/10
        #set pixel = 0 if there is a difference and =1 if there is not a big enough difference
        #to be an edge
        for i in range(col):
            for j in range(row-1):
                if abs(Filter[j][i][0] - Filter[j+1][i][0] > mindiff):
                    image[j][i][0] = 1
                    image[j][i][1] = 1
                    image[j][i][2] = 1
                else:
                    image[j][i][0] = 0
                    image[j][i][1] = 0
                    image[j][i][2] = 0
        #for the last row of pixels, must compare above pixel instead of below
        for i in range(col):
            if abs(Filter[row-1][i][0] - Filter[row-2][i][0] > mindiff):
                image[row-1][i][0] = 1
                image[row-1][i][1] = 1
                image[row-1][i][2] = 1
            else:
                image[row-1][i][0] = 0
                image[row-1][i][1] = 0
                image[row-1][i][2] = 0
       
                
                
    if option == 'v':
        Filter = np.array([[[0.0,0.0,0.0] for _ in range(row)] for _ in range(col)])
        F = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
        #multiply kxk box by filter F and find value of pixel i, j (pix)
        for i in range(row):
            for j in range(col):
                #make sure not to include edges
                if i >= floor(3/2) and i <= row-ceil(3/2) and j >= floor(3/2) and j <= col - ceil(3/2):
                    pix = 0
                    r = -1
                    for m in range(i-floor(3/2), i+ceil(3/2)):
                        r += 1
                        c = -1
                        for n in range(j-floor(3/2), j+ceil(3/2)):
                            c+= 1
                            pix += F[r][c]*image[m][n][0]
                    Filter[i][j][0]= pix
                    Filter[i][j][1]= pix
                    Filter[i][j][2]= pix 
        #find max difference betwen pixel differences
        maxdiff = 0
        for i in range(row):
            for j in range(col-1):
                if abs(Filter[i][j][0] - Filter[i][j+1][0] > maxdiff):
                    maxdiff = abs(Filter[i][j][0] - Filter[i][j+1][0])
        #set an arbitrary boundary of what the amount of the minimum difference 
        #between 2 pixels that constitute an edge is
        mindiff = maxdiff/10
        #set pixel = 0 if there is a difference and =1 if there is not a big enough difference
        #to be an edge
        for i in range(row):
            for j in range(col-1):
                if abs(Filter[i][j][0] - Filter[i][j+1][0] > mindiff):
                    image[i][j][0] = 1
                    image[i][j][1] = 1
                    image[i][j][2] = 1
                else:
                    image[i][j][0] = 0
                    image[i][j][1] = 0
                    image[i][j][2] = 0
        #for the last row of pixels, must compare above pixel instead of below
        for i in range(row):
            if abs(Filter[i][col-1][0] - Filter[i][col-2][0] > mindiff):
                image[i][col-1][0] = 1
                image[i][col-1][1] = 1
                image[i][col-1][2] = 1
            else:
                image[i][col-1][0] = 0
                image[i][col-1][1] = 0
                image[i][col-1][2] = 0
    if option == 'b':
        Filter = np.array([[[0.0,0.0,0.0] for _ in range(row)] for _ in range(col)])
        Fx = np.array([[1,2,1], [0,0,0],[-1,-2,-1]])
        Fy = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
        #multiply kxk box by filter F and find value of pixel i, j (pix)
        for i in range(row):
            for j in range(col):
                #make sure not to include edges
                if i >= floor(3/2) and i <= row-ceil(3/2) and j >= floor(3/2) and j <= col - ceil(3/2):
                    Gx = 0
                    Gy = 0
                    r = -1
                    for m in range(i-floor(3/2), i+ceil(3/2)):
                        r += 1
                        c = -1
                        for n in range(j-floor(3/2), j+ceil(3/2)):
                            c+= 1
                            Gx += Fx[r][c]*image[m][n][0]
                    r = -1
                    for m in range(i-floor(3/2), i+ceil(3/2)):
                        r += 1
                        c = -1
                        for n in range(j-floor(3/2), j+ceil(3/2)):
                            c+= 1
                            Gy += Fy[r][c]*image[m][n][0]
                    pix = sqrt(Gx**2 + Gy**2)
                    Filter[i][j][0]= pix
                    Filter[i][j][1]= pix
                    Filter[i][j][2]= pix 
        return Filter
    return image
 
from PIL import Image
import os
img = Image.open('Cat.jpg').convert('L')
img.save('greyscale.jpg')
nim = np.array(Image.open('greyscale.jpg'))

###splits a gray-scale image into foreground and background using
###Otsu’s thresh-h￼￼olding method
###input:
###@image: a gray-scale picture
###@output:
###the image split into foreground and background
def otsu_threshold(image):
    nim = np.array(image)
    row = nim.shape[0]
    col = nim.shape[1]
    hist,bins  = np.histogram(nim.ravel(), 256, [0,256])
    #find mean value of hist
    m = 0#the mean
    for i in range(len(hist)):
        m += hist[i]*i
    m = m/(row*col)
    
    maxV = 0
    T = 0
    #find threshold T through comparing inter-class variances at different
    #points t and maximizing the value
    for i in range(1, len(hist)):
        t = i#threshold to test
        Fg = 0#set of pixels with color <= t
        Bg = 0#set of pixels with color > t
        mFg = 0#mean color of Fg
        mBg = 0#mean color of Bg
        denom = 0
        var = 0
        for k in range(i):
            Fg += hist[k]
            mFg += hist[k]*k
            denom += hist[k]
        mFg = mFg/(row*col)
        denom2 = 0
        for n in range(len(hist)-i):
            Bg += hist[n]
            mBg += hist[n]*n
            denom2 += hist[n]
        mBg = mBg/(row*col)
        #calculate inter-class variance
        var = ((Fg/(row*col))*(mFg - m)**2) + ((Bg/(row*col))*(mBg - m)**2)
        if var > maxV:
            maxV = var
            T = t
    #all pixels less than threshold is set to 0 and all others are set to 255
    for i in range(row):
        for j in range(col):
            if nim[i][j] < T:
                nim[i][j] = 0
            else:
                nim[i][j] = 255 
    #create Image object
    otsuIm = Image.fromarray(nim)
    return otsuIm

###identifies the background of an image and blurs it
###input:
###@image: a gray-scale picture
###@output:
###the image with blurred background
def blur_background(image):
    otsu = otsu_threshold(image)
    notsu = np.array(otsu)
    row = notsu.shape[0]
    col = notsu.shape[1]
    image = np.array(image)
    update = np.array([[0.0 for _ in range(col)] for _ in range(row)])
    k = 40
    
    #calculate number of black and white pixels, the one w/ greater # is the background
    black = 0
    white = 0
    for i in range(row):
        for j in range(col):
            if notsu[i][j] == 255:
                black += 1
            if notsu[i][j] == 0:
                white += 1
    back = 0
    if black > white:
        back = 255
    for i in range(row):
            for j in range(col):
                #make sure not to include edges
                if i >= floor(k/2) and i <= row-ceil(k/2) and j >= floor(k/2) and j <= col - ceil(k/2):
                    #if part of the background blur it
                    if notsu[i][j] == back:
                        
                        sum = 0
                        #average the pixels around i,j of k by k square
                        for m in range(i-floor(k/2), i+ceil(k/2)):
                            for n in range(j-floor(k/2), j+ceil(k/2)):
                                sum += image[m][n]/k**2
                        update[i][j] = sum
                        
                    else:
                        update[i][j] = image[i][j]
                else:
                    update[i][j] = image[i][j]
    #create image object
    updateIm = Image.fromarray(update)
    return updateIm
    
