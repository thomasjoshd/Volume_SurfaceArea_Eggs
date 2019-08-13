#Author: Dr. Joshua Thomas
#thomas.joshd@gmail.com
#Clarkson University

UPDATED="21-JUN-2019"
#Version 9 switch to class structure, and make compatible with python 3
#Version 10 adds the scale bar feature of the section modulus program
version=10
#This program calculates the volume and surface area of spheres and oblate objects
#from a 2-D iamge assuming symmetric about long axis.

import cv2 #image library
import numpy as np #arrays and math
import matplotlib.pyplot as plt #basic plotting
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse #needed for Ben Hammel's code

try:
    #Python 2.7 call
    import Tkinter as tk   #GUI
    from tkFileDialog   import askopenfilename
    import tkMessageBox as messagebox
except:
    #Python 3 call
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    from tkinter import messagebox

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as NavigationToolbar2Tk
except:
    pass
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except:
    pass

print("This Software requires the following python libraries: opencv2, numpy, matplotlib, and Tkinter")

#######################################
#MIT License

#Copyright (c) 2018 Ben Hammel

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


#import numpy  #I modified the function calls to match how numpy etc were imported above.
#import matplotlib.pyplot as plt
#from matplotlib.patches import Ellipse

"""Demonstration of least-squares fitting of ellipses
    __author__ = "Ben Hammel, Nick Sullivan-Molina"
    __credits__ = ["Ben Hammel", "Nick Sullivan-Molina"]
    __maintainer__ = "Ben Hammel"
    __email__ = "bdhammel@gmail.com"
    __status__ = "Development"
    Requirements
    ------------
    Python 2.X or 3.X
    numpy
    matplotlib
    References
    ----------
    (*) Halir, R., Flusser, J.: 'Numerically Stable Direct Least Squares
        Fitting of Ellipses'
    (**) http://mathworld.wolfram.com/Ellipse.html
    (***) White, A. McHale, B. 'Faraday rotation data analysis with least-squares
        elliptical fitting'
"""

class LSqEllipse:

    def fit(self, data):
        """Lest Squares fitting algorithm
        Theory taken from (*)
        Solving equation Sa=lCa. with a = |a b c d f g> and a1 = |a b c>
            a2 = |d f g>
        Args
        ----
        data (list:list:float): list of two lists containing the x and y data of the
            ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
        Returns
        ------
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
        """
        x, y = np.asarray(data, dtype=float)

        #Quadratic part of design matrix [eqn. 15] from (*)
        D1 = np.mat(np.vstack([x**2, x*y, y**2])).T
        #Linear part of design matrix [eqn. 16] from (*)
        D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T

        #forming scatter matrix [eqn. 17] from (*)
        S1 = D1.T*D1
        S2 = D1.T*D2
        S3 = D2.T*D2

        #Constraint matrix [eqn. 18]
        C1 = np.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

        #Reduced scatter matrix [eqn. 29]
        M=C1.I*(S1-S2*S3.I*S2.T)

        #M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this equation [eqn. 28]
        eval, evec = np.linalg.eig(M)

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4*np.multiply(evec[0, :], evec[2, :]) - np.power(evec[1, :], 2)
        a1 = evec[:, np.nonzero(cond.A > 0)[1]]

        #|d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -S3.I*S2.T*a1

        # eigenvectors |a b c d f g>
        self.coef = np.vstack([a1, a2])
        self._save_parameters()


    def _save_parameters(self):
        """finds the important parameters of the fitted ellipse

        Theory taken form http://mathworld.wolfram
        Args
        -----
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
        Returns
        _______
        center (List): of the form [x0, y0]
        width (float): major axis
        height (float): minor axis
        phi (float): rotation of major axis form the x-axis in radians
        """

        #eigenvectors are the coefficients of an ellipse in general form
        #a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
        a = self.coef[0,0]
        b = self.coef[1,0]/2.
        c = self.coef[2,0]
        d = self.coef[3,0]/2.
        f = self.coef[4,0]/2.
        g = self.coef[5,0]

        #finding center of ellipse [eqn.19 and 20] from (**)
        x0 = (c*d-b*f)/(b**2.-a*c)
        y0 = (a*f-b*d)/(b**2.-a*c)

        #Find the semi-axes lengths [eqn. 21 and 22] from (**)
        numerator = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        denominator1 = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        denominator2 = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        width = np.sqrt(numerator/denominator1)
        height = np.sqrt(numerator/denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)
        # or [eqn. 26] from (***).
        phi = .5*np.arctan((2.*b)/(a-c))

        self._center = [x0, y0]
        self._width = width
        self._height = height
        self._phi = phi

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def phi(self):
        """angle of counterclockwise rotation of major-axis of ellipse to x-axis
        [eqn. 23] from (**)
        """
        return self._phi

    def parameters(self):
        return self.center, self.width, self.height, self.phi


def make_test_ellipse(center=[1,1], width=1, height=.6, phi=3.14/5):
    """Generate Elliptical data with noise

    Args
    ----
    center (list:float): (<x_location>, <y_location>)
    width (float): semimajor axis. Horizontal dimension of the ellipse (**)
    height (float): semiminor axis. Vertical dimension of the ellipse (**)
    phi (float:radians): tilt of the ellipse, the angle the semimajor axis
        makes with the x-axis
    Returns
    -------
    data (list:list:float): list of two lists containing the x and y data of the
        ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
    """
    t = numpy.linspace(0, 2*numpy.pi, 1000)
    x_noise, y_noise = numpy.random.rand(2, len(t))

    ellipse_x = center[0] + width*numpy.cos(t)*numpy.cos(phi)-height*numpy.sin(t)*numpy.sin(phi) + x_noise/2.
    ellipse_y = center[1] + width*numpy.cos(t)*numpy.sin(phi)+height*numpy.sin(t)*numpy.cos(phi) + y_noise/2.

    return [ellipse_x, ellipse_y]



####END Code by Ben Hammel#############
#######################################


class App:
    def __init__(self,master):
        self.master=master
        master.title("Volume and Surface Area, Version %s"%str(version))

        #input for blurring the image, aids in edge detection for low resolution images
        tk.Label(self.master, text="Parameters for Automatic Edge Detection").pack(side=tk.TOP)
        A1 = tk.Label(self.master, text="Edge blurring number of pixels").pack(side=tk.TOP)
        self.vv=tk.IntVar()
        self.B1=tk.Entry(self.master,text=self.vv)
        self.B1.pack(side = tk.TOP)
        self.vv.set(5)

        #these edges are an integer between 0 and 255 that control how the edges are found, helpful in low contrast.
        self.w1L = tk.Label(self.master, text="Edge finding lower limit").pack(side=tk.TOP)
        self.w1v=tk.DoubleVar()
        self.w1 = tk.Entry(self.master,text=self.w1v)
        self.w1.pack(side = tk.TOP)
        self.w1v.set(30)    #default value of so the program doesn't barf if accidentally pushed.

        self.w2L = tk.Label(self.master, text="Edge finding upper limit").pack(side=tk.TOP)
        self.w2v=tk.DoubleVar()
        self.w2 = tk.Entry(self.master,text=self.w2v)
        self.w2.pack(side = tk.TOP)
        self.w2v.set(100)   #default value of so the program doesn't barf if accidentally pushed.


        #this is needed to put the calculations in real units, the length of the longest dimension of the egg.
        L1 = tk.Label(self.master, text="Longest Length of Object, answers are in the base unit used here.").pack(side=tk.TOP)
        self.len=tk.IntVar()
        self.e1 = tk.Entry(self.master,text=self.len)
        self.e1.pack(side = tk.TOP)
        self.len.set(1)   #default value of so the program doesn't barf if accidentally pushed.

        #input for polynomial order fit to edges, need to have a smooth edge to make a good approximation of the surface area.
        D1 = tk.Label(self.master, text="Order for polynomial fit").pack(side=tk.TOP)
        self.vvv=tk.IntVar()
        self.C1=tk.Entry(self.master,text=self.vvv)
        self.C1.pack(side = tk.TOP)
        self.vvv.set(16) #default order for polynomial

        #Output boxes for the calculations
        self.output=tk.Entry(self.master,width=50)
        self.output.pack(side=tk.TOP)
        self.output2=tk.Entry(self.master,width=50)
        self.output2.pack(side=tk.TOP)


        #file menu etc
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)
        filemenu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open File (o)", command=self.OpenFile)
        filemenu.add_command(label="Reset", command=self.Reset)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self._quit)

        barmenu = tk.Menu(menu)
        menu.add_cascade(label="Scale Bar", menu=barmenu)
        barmenu.add_command(label="Only use/need these options if there is a scale bar.")
        barmenu.add_command(label="Set Background Color", command=self.background)
        barmenu.add_command(label="Click Image Scale Bar",command=self.ScaleBarClicks)


        edgemenu = tk.Menu(menu)
        menu.add_cascade(label="Edge Find", menu=edgemenu)
        edgemenu.add_command(label="Automatic Edge detection", command=self.find_edges)
        edgemenu.add_separator()
        edgemenu.add_command(label="Manually click along the edges of the egg", command=self.ManualOverride)

        calcmenu = tk.Menu(menu)
        menu.add_cascade(label="Calculate Volume & Surf. Area", menu=calcmenu)
        calcmenu.add_command(label="Calculate", command=self.Calculate)
        calcmenu.add_command(label="Display Results", command=self.Display)

        helpmenu = tk.Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...", command=self.About)

        self.generate_plot()
        self.clicks_check=0
        self.edges_check=0
        self.master.bind('o', self.OpenFile)


    def OpenFile(self,event=None):
        self.Reset()
        self.fname = askopenfilename() #file dialog
        self.img = cv2.imread(self.fname) #read image
        self.implot()
        # horizontal=int(float(np.shape(self.img)[0])/(100/float(np.shape(self.img)[1])))
        # vertical=int(float(np.shape(self.img)[1])/(100/float(np.shape(self.img)[1])))
        # self.img=mpimg.imread(self.fname)
        # self.ax.clear()
        # self.ax.imshow(self.img)
        # self.ax.set_title("Quick display to make sure you've selected the correct image.")
        # self.canvas.draw()
        print("----------------------------------")
        print("Filename : %s" %self.fname)




    def implot(self):
        self.ax.clear()
        self.ax.set_title("%s"%(self.fname),fontsize=10)
        self.ax.imshow(self.img)
        self.toolbar.update()
        self.canvas.draw()

    def generate_plot(self):
        self.fig=plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.tick_params(right= False,top= False,left= False, bottom= False, labelbottom=False,labelleft=False )
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk( self.canvas, self.master )
        self.toolbar.update()
        self.canvas.draw()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


    def ManualOverride(self):
        self.clicks=None
        self.edges=None
        try:
            self.img = cv2.imread(self.fname) #read the image
        except:
            self.fname=askopenfilename() #if a file hasn't already been opened, do it now.
        self.iimg = cv2.imread(self.fname)
        print("Filename : %s" %self.fname)
        self.iimg=mpimg.imread(self.fname)
        #use ginput to manually get the edges of the egg.
        self.ax.clear()
        self.ax.imshow(self.iimg)
        print("Please Left click to add a point.  Right click to remove a point.  Middle Click to end.")
        self.ax.set_title("Left Click: add a location the the array of the egg edges \n Right Click: Remove last added point. ")
        self.ax.set_xlabel("Middle Click (or both left and right): Finished selecting the edge of the egg.")
        self.canvas.draw()
        self.clicks = np.array(plt.ginput(0))
        self.ax.plot(self.clicks[:,0],self.clicks[:,1])
        self.canvas.draw()
        self.clicks_check=1

    def find_edges(self):
        self.edges=None
        self.clicks=None
    #displays automatically found edges
        try:
            del self.clicks
        except:
            pass

        # try:
        #     self.img = cv2.imread(self.fname) #read the image
        # except:
        #     self.fname=askopenfilename() #if a file hasn't already been opened, do it now.
        #     self.img = cv2.imread(self.fname)
        #     print("Filename : %s" %self.fname)

        blur = cv2.blur(self.img,(int(self.B1.get()),int(self.B1.get())))
        self.edges = cv2.Canny(blur,float(self.w1.get()),float(self.w2.get())) #finds the edges, the w1 and w2 come from the sliders.  w1 should be lower than w2.
        self.ax.clear()
        self.ax.imshow(self.edges) #show the image and contours
        self.ax.set_title("Edges should be yellow and only along the outside of the egg. \n Adjust the automatic edge, and blurring parameters if needed and re-run.")
        self.canvas.draw()
        self.edges_check=1


    # def eigenvec(self,xe,ye):
    #     #use principal component analysis to find the major and minor axis of the object.
    #     x_m=np.mean(xe) #approximate the center
    #     y_m=np.mean(ye)
    #     xc=xe - x_m #move the orgin to the center of the object
    #     yc=ye - y_m
    #     coords = np.vstack([xc,yc])
    #     cov = np.cov(coords)
    #     evals, evecs = np.linalg.eig(cov) #find the eigenvectors for PCA
    #     sort_indices = np.argsort(evals)[::-1]
    #     evec1, evec2 = evecs[:, sort_indices]
    #     return evec1, evec2,coords


    def show_image(self): #show the image
        try:
          self.img = cv2.imread(self.fname) #read the image
        except:
          self.fname=askopenfilename() #if a file hasn't already been opened, do it now.
          self.img = cv2.imread(self.fname)
        print("Filename : %s" %self.fname)
        self.ax.imshow(self.img, origin='lower') #show the image and contours
        self.canvas.draw()

    def LocalRadius(self,A,B,C):
        #Find vectors connecting the three points and the length of each vector
        AB = B - A
        BC = C - B
        AC = C - A
        # Triangle Lengths
        a = np.linalg.norm(AB)
        b = np.linalg.norm(BC)
        c = np.linalg.norm(AC)
        return (a * b * c) / np.sqrt(2.0 * a**2 * b**2 + 2.0 * b**2 * c**2 + 2.0 * c**2 * a**2 - a**4 - b**4 - c**4)

    def edge2array(self):
        ans=[]  #turn our edges into x,y points
        for y in range(0, self.edges.shape[0]):
            for x in range(0, self.edges.shape[1]):
                if self.edges[y, x] != 0:
                    ans = ans + [[x, y]]
        ans = np.array(ans)
        return ans

    def pltregion(self,x,y,sym='-',c='red'):
        #takes two clicks
        self.ax.plot(x,y,sym,color=c)
        self.canvas.draw()

    def region(self):
        #uses two clicks to define a region for fitting or measuring.
        self.output.delete(0,tk.END)
        self.output.insert(tk.END,"Left Click (x) on the ends of the scale bar. Right Click (backspace) to remove a point.")
        clicks= plt.ginput(2)
        clicks= np.array(clicks)
        x=[]
        y=[]
        for i,j in enumerate(clicks):
            x.append(clicks[i][0])
            y.append(clicks[i][1])
        self.pltregion(x,y,sym='x',c='red')
        self.ignore()
        return x,y

    def ScaleBarClicks(self):
        #manually click on a scale bar
        x,y=self.region()
        self.scale=int(max([abs(x[1]-x[0]),abs(y[1]-y[0])]))

    def background(self):
        #use a single click to get background color
        self.output.delete(0,tk.END)
        self.output.insert(tk.END,"Left Click (x) on a point containing uniform background")
        clicks= plt.ginput(1)
        x,y=int(clicks[0][0]),int(clicks[0][1])
        self.bgcolor=self.img[y,x]
        endmess=", Scale bar or Find bone next."
        if self.bgcolor[0] == 0:
            message="Background is probably Black"+endmess
            print(message)
            self.output.delete(0,tk.END)
            self.output.insert(tk.END,message)
            self.bglabel='black'
        elif self.bgcolor[0] > 200:
            message="Background is probably White"+endmess
            print(message)
            self.output.delete(0,tk.END)
            self.output.insert(tk.END,message)
            self.bglabel='white'
        else:
            message="Background selection failure"+endmess
            print(message)
            self.output.delete(0,tk.END)
            self.output.insert(tk.END,message)

    def ignore(self):
        #define a box to exclude 4 points
        self.output.delete(0,tk.END)
        self.output.insert(tk.END,"Click on corners of a box to exclude. Right Click (backspace) to remove a point.")
        clicks= plt.ginput(4)
        clicks= np.array(clicks)
        x=[]
        y=[]
        for i,j in enumerate(clicks):
            x.append(clicks[i][0])
            y.append(clicks[i][1])
        self.pltregion(x,y,sym='s',c='blue')
##        print(np.shape(self.img))
##        print(min(x),max(x),min(y),max(y))
        xbox=np.arange(min(x),max(x),dtype=int)
        ybox=np.arange(min(y),max(y),dtype=int)
##        print(xbox,ybox)
        for xidx in xbox:
            for yidx in ybox:
                self.img[yidx,xidx]=self.bgcolor
        self.implot()


    def eigenvec(self):
        coords = np.vstack([self.xc,self.yc])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov) #find the eigenvectors for PCA
        sort_indices = np.argsort(evals)[::-1]
        evec1, evec2 = evecs[:, sort_indices]
        return evec1, evec2,coords

    def get_longestlength(self):
        #Find longest axis with principle componenet analysis
##        global contours

        evec1, evec2,coords = self.eigenvec()
        x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evec2

        #rotate a copy of the object
        phi=-1*np.arctan2(y_v1,x_v1)+np.pi/2.
        #print("Phi = ",phi)
        xr=self.xc*np.cos(phi)-self.yc*np.sin(phi)
        yr=self.yc*np.cos(phi)+self.xc*np.sin(phi)
        xr_v1=x_v1*np.cos(phi)-y_v1*np.sin(phi)  #really only need rotated vector
        yr_v1=y_v1*np.cos(phi)+x_v1*np.sin(phi)

        #plot the stuff
##        self.ax.clear()
        self.diameter = max(yr)+abs(min(yr))+1 #the plus to count the centero pixel

        #plt.plot([x_v2*-scale, x_v2*scale], [y_v2*-scale, y_v2*scale], color='blue') #plot minor axis
##        self.ax.plot(xr, yr, '.',label="Detected Material")
        #plt.plot(0,0,'s') #plot center
        self.ax.plot([x_v1*-self.diameter/2, x_v1*self.diameter/2], [y_v1*-self.diameter/2, y_v1*self.diameter/2], color='red') #plot major axis
##        self.ax.plot([xr_v1*-self.scale/2, xr_v1*self.scale/2], [yr_v1*-self.scale/2, yr_v1*self.scale/2], color='red') #plot major axis

        self.ax.axis('equal') #square aspect ratio, so cirles look like circles
##        self.ax.invert_yaxis()  # Match the image system with origin at top left
        self.ax.set_title("Cross-section, red axis is 'longest length'")
        self.toolbar.update()
        self.ax.legend()
        self.canvas.draw()


    def Calculate(self):
    #Find longest axis with principle componenet analysis, rotate and use solids and surfaces of revolution to calculate
    #surface area with the idea of solids of revolution.
        if self.clicks_check == 1:
            ans=self.clicks
        elif self.edges_check == 1:
            ans=self.edge2array()
        else:
            print("Please select an image first!")
            messagebox.showinfo("No Image","No edges have been found.\nPlease select an image first!")

        x=ans[:,0] #x values
        y=ans[:,1] #y values
        xo=ans[:,0]
        yo=ans[:,1]

        #fit ellipse to edges
        lsqe=LSqEllipse()
        lsqe.fit([x,y])
        center,width,height,phi=lsqe.parameters()
        print("phi=",phi)
        #move center to origin
        x=xo-center[0] #move the orgin to the center of the object
        y=yo-center[1]
        #rotate
        self.xc=x*np.cos(-phi)-y*np.sin(-phi)
        self.yc=y*np.cos(-phi)+x*np.sin(-phi)
        #
        # evec1, evec2,coords = self.eigenvec()
        # x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
        # x_v2, y_v2 = evec2

        #Convert pixels to real length units, whatever the units are that you enter in the GUI are used
        egglength=float(self.e1.get()) #get length of longest axis from a real measurement
        # pixellength=max(yc)-min(yc)

        self.get_longestlength()
        if self.scale == 0:
            pixellength=self.diameter
        else:
            pixellength=self.scale

        eggscale=egglength/pixellength

        #Compute the volume and area, find radius from symmetry axis
        vol=[] #empty volume list
        vol2=[]
        ar2=[]
        ar=[]

        # Build 2-d array, and fill it, then split along symmetry axis.  Allows us to use the left and right halves separately.
        XY=np.stack((self.xc,self.yc),axis=-1)
        Cleft=[]
        Cright=[]
        for C in XY:
            if C[0] < 0:
                Cleft.append(C)
            else:
                Cright.append(C)

        CL=np.array(Cleft)
        indL=np.lexsort((CL[:,0],CL[:,1]))  #sort the coordinates so that they from curve that can be fit by a function.
        CR=np.array(Cright)
        indR=np.lexsort((CR[:,0],CR[:,1]))
        CLs=CL[indL]
        CRs=CR[indR]

        CRz=np.polyfit(CRs[:,1],CRs[:,0],int(self.C1.get())) #fit y, then x
        CRp=np.poly1d(CRz)
        CLz=np.polyfit(CLs[:,1],CLs[:,0],int(self.C1.get())) #fit y, then x
        CLp=np.poly1d(CLz)

        yp=np.linspace(min(CRs[:,1]),max(CRs[:,1]),1000)  #uniform sampling along the fit to the edges.

        #plot the found edges and the polynomial fits to them.
        self.ax.clear()
        self.ax.plot(CRs[:,0], -CRs[:,1], 'bs',label="Found Edge Right")
        self.ax.plot(CRp(yp), -yp, 'r-',label="Polynomial Fit Right")
        self.ax.plot(CLs[:,0], -CLs[:,1], 'rs', label="Found Edge Left")
        self.ax.plot(CLp(yp), -yp, 'b-',label="Polynomial Fit Left")
        self.ax.set_title("Check that the curves are smooth and match the edge of egg.")
        self.ax.set_xlabel("If the curves look good you're done.  Otherwise change the fit order or reselect the edges.")
        self.ax.axis('equal')
        self.ax.legend()
        self.ax.set_ylim(min(self.yc),max(self.yc))
        self.canvas.draw()

        #Compute Volume and  Area
        radius=[]
        for i,val in enumerate(yp):
            if i == 0:
                pass
            else:
                r1=abs(CLp(yp[i]))
                dy=abs(yp[i]-yp[i-1])  #common to both.
                dx1=abs(CLp(yp[i])-CLp(yp[i-1]))
                vol.append(r1**2*dy) #area of a disk, times thickness = volume
                ar.append(r1*np.sqrt(dx1**2+dy**2)) #circumfrence of disk times thickness = area
                radius.append(r1*eggscale)
                r2=abs(CRp(yp[i]))
                dx2=abs(CRp(yp[i])-CRp(yp[i-1]))
                vol2.append(r2**2*dy) #area of a disk, times thickness = volume
                ar2.append(r2*np.sqrt(dx2**2+dy**2)) #circumfrence of disk times thickness = area

        #compute an average and std for both sides, yay science!
        self.v=sum(vol)*np.pi*eggscale**3
        self.v2=sum(vol2)*np.pi*eggscale**3
        self.a=sum(ar)*2.*np.pi*eggscale**2
        self.a2=sum(ar2)*2.*np.pi*eggscale**2
        self.ave_volume=np.average([self.v,self.v2])
        self.ave_area=np.average([self.a,self.a2])
        self.std_volume=np.std([self.v,self.v2])
        self.std_area=np.std([self.a,self.a2])
        self.output.delete(0,tk.END)
        self.output.insert(tk.END,"Ave. Volume: %.3f, STD: %.3f"%(self.ave_volume,self.std_volume))
        self.output2.delete(0,tk.END)
        self.output2.insert(tk.END,"Ave. Area: %.3f, STD: %.3f"%(self.ave_area,self.std_area))


    def Display(self):
        print("----------------------------------")
        print("Filename : %s" %self.fname)
        print("Volume using left side as symmetric: %s" %self.v)
        print("Volume using right side as symmetric: %s" %self.v2)
        print("Area using left side as symmetric: %s" %self.a)
        print("Area using right side as symmetric: %s" %self.a2)
        print("Average Volume: %s, STD: %s"%(self.ave_volume,self.std_volume))
        print("Average Surface Area: %s, STD: %s"%(self.ave_area,self.std_area))
        print("The two sides are the calculations based on radii on either side of the symmetry axis.  The average and standard deviation are calculated based on these values.")
        print("----------------------------------")

        t=tk.Toplevel(self.master,height=600,width=600)
        t.wm_title("Detailed Output")
        s=tk.Scrollbar(t)
        s.pack(side=tk.RIGHT,fill=tk.Y)
        hlist=tk.Listbox(t,yscrollcommand=s.set,height=20,width=80)
        hlist.insert(tk.END, "Filename : %s" %self.fname)
        hlist.insert(tk.END, "Volume using left side as symmetric: %s" %self.v)
        hlist.insert(tk.END, "Volume using right side as symmetric: %s" %self.v2)
        hlist.insert(tk.END, "Area using left side as symmetric: %s" %self.a)
        hlist.insert(tk.END, "Area using right side as symmetric: %s" %self.a2)
        hlist.insert(tk.END, "Average Volume: %s, STD: %s"%(self.ave_volume,self.std_volume))
        hlist.insert(tk.END, "Average Surface Area: %s, STD: %s"%(self.ave_area,self.std_area))
        hlist.insert(tk.END, "The two sides are the calculations based on radii on either side of the symmetry axis.")
        hlist.insert(tk.END, "The average and standard deviation are calculated based on these values.")
        hlist.pack(side=tk.LEFT,fill=tk.BOTH)
        s.config(command=hlist.yview)


    def About(self):
        t=tk.Toplevel(self.master,height=600,width=60)
        t.wm_title("About")
        tk.Label(t,text="Quick start steps: Open image, find edges, calculate, display").pack()
        tk.Label(t,text=" ").pack()
        tk.Label(t,text="Basic Volume and Surface Area calculation from a cross-section.").pack()
        tk.Label(t,text="The program fits a 16th order polynomial to the found edge to better calculate surface area.").pack()
        tk.Label(t,text="Includes Ellipse fitting by Ben Hammel, see source code for details.").pack()
        tk.Label(t,text=" ").pack()
        tk.Label(t,text="Author: Dr. Joshua Thomas \n thomas.joshd@gmail.com").pack()
        tk.Label(t,text="Version %s"%version).pack()
        tk.Label(t,text="Last Updated %s"%UPDATED).pack()

    def _quit(self):
        self.master.destroy()
        self.master.quit()

    def Reset(self):
        self.output.delete(0,tk.END)
        self.output2.delete(0,tk.END)
        self.vv.set(5)
        self.w1v.set(30)
        self.w2v.set(100)
        self.len.set(1)
        self.vvv.set(16)
        self.clicks=None
        self.edges=None
        self.clicks_check=0
        self.edges_check=0
        self.v=None
        self.v2=None
        self.a=None
        self.a2=None
        self.ave_volume=None
        self.ave_area=None
        self.std_volume=None
        self.std_area=None
        self.fname=None
        self.ax.clear()
        self.canvas.draw()
        self.scale=0


#----------------------------------------------------------------------------------
#Begin Grafical User Interface
root = tk.Tk() #main GUI window
program=App(root)
root.protocol("WM_DELETE_WINDOW", program._quit)
root.mainloop() #lets the GUI run
