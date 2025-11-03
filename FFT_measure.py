#%% GUI building via Tkinter 
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import cv2
import numpy as np  



#%%
class App:
    # set the UI interface
    def __init__(self,root):
        self.root = root # initialize the root window
        self.root.title("FFT measurement") # title of the root window
        self.root.geometry("1500x1000") # size of the root window
        self.root.minsize(800,600) # minimum size of the root window


        # initial image and canvas view parameters
        self.img_pil = None   # PIL image object
        self.fft_pil = None   # FFT PIL image object
        self.gray_array = None # grayscale numpy array
        self.gray_fft = None
        self.tk_image = None  # Tkinter image object
        self.scale = 1.0      # sclae = 0, original size of the image
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.offx = 0         # image top-left
        self.offy = 0
        self.px = None
        self.rpx = None
        self.dist_from_fft = None
        self.fft_line_coor = None

        self._set_widgets() #  set the widgets
    
    # function to set the widgets, run automatically when instantiated the class
    def _set_widgets(self):
        
        # control frame
        self.control_frame = ttk.Frame(self.root,padding=10,relief='ridge') # control frame

        # image_fft frame
        self.image_fft_frame = ttk.Frame(self.root,padding=10,relief='ridge') # image, fft frame, with internel padding of 10

        # line profile frame
        self.lp_frame = ttk.Frame(self.root,padding=10,relief='ridge') # lineprofile frame

        self.control_frame.pack(side='left',fill='both',expand=1,padx=10,pady=10) 
        self.image_fft_frame.pack(side='left',fill='both',expand=1,padx=10,pady=10) 
        #self.lp_frame.pack(side='left',fill='both',expand=1,padx=10,pady=10) 

   

        ######################## Image load button #####################
        self.control_label = ttk.Label(self.control_frame,text="Control panel") #create a hello button in the control frame
        self.control_label.pack() # how the button is packed in the control_frame
        
        self.load_image_btn = ttk.Button(self.control_frame,text="Load Image",command=self._load_display_image) #create a hello button in the control frame
        self.load_image_btn.pack(pady=10) 

        self.px_input_frame = ttk.Frame(self.control_frame,padding=2)
        self.px_input_frame.pack(side='top',pady=5)

        self.px_input_label = ttk.Label(self.px_input_frame,text='Pixel size:')
        self.px_input_label.pack(side='left')

        entry_var = tk.StringVar(value='')
        self.px_input_box = ttk.Entry(self.px_input_frame,textvariable=entry_var,width=5)
        self.px_input_box.pack(side='left')

        self.px_input_unit = ttk.Label(self.px_input_frame,text = "Å")
        self.px_input_unit.pack(side='left',padx = 3)

        ######################## Brightness slide bar #####################
        self.img_bright_slide_frame = ttk.Frame(self.control_frame,padding=2)
        self.img_bright_slide_frame.pack(side = 'top',pady=5,fill='x')

        self.img_brightslide_label = ttk.Label(self.img_bright_slide_frame,text='Brightness')
        self.img_brightslide_label.pack(side='left',expand=1,fill='x')

        self.img_brightness_value = tk.DoubleVar(value=50)
        self.img_bright_slidebar = ttk.Scale(self.img_bright_slide_frame,from_=0,to=100,variable=self.img_brightness_value,
                                             command=self._change_image)
        self.img_bright_slidebar.pack(side='left')

        self.img_bright_reset = ttk.Button(self.img_bright_slide_frame,text='reset',command=self._reset_br_img)
        self.img_bright_reset.pack(side='left')

     
        ################# Contrast slide bar###################
        self.img_contrast_slide_frame = ttk.Frame(self.control_frame,padding=2)
        self.img_contrast_slide_frame.pack(side = 'top',pady=5,fill='x')

        self.img_contrastslide_label = ttk.Label(self.img_contrast_slide_frame,text='Contrast')
        self.img_contrastslide_label.pack(side='left',expand=1,fill='x')

        self.img_contrast_value = tk.DoubleVar(value=50)
        self.img_contrast_slidebar = ttk.Scale(self.img_contrast_slide_frame,from_=0,to=100,variable=self.img_contrast_value,
                                               command=self._change_image)
        self.img_contrast_slidebar.pack(side='left')

        self.img_contrast_reset = ttk.Button(self.img_contrast_slide_frame,text='reset',command=self._reset_ct_img)
        self.img_contrast_reset.pack(side='left')

        ################# alpha slide bar ###################
        self.img_gamma_slide_frame = ttk.Frame(self.control_frame,padding=2)
        self.img_gamma_slide_frame.pack(side = 'top',pady=5,fill='x')

        self.img_gammaslide_label = ttk.Label(self.img_gamma_slide_frame,text='Gamma')
        self.img_gammaslide_label.pack(side='left',expand=1,fill='x')

        self.img_gamma_value = tk.DoubleVar(value=50)
        self.img_gamma_slidebar = ttk.Scale(self.img_gamma_slide_frame,from_=0,to=100,variable=self.img_gamma_value,
                                            command=self._change_image)
        self.img_gamma_slidebar.pack(side='left')

        self.img_gamma_reset = ttk.Button(self.img_gamma_slide_frame,text='reset',command=self._reset_ga_img)
        self.img_gamma_reset.pack(side='left')

        ####################### FFT calculation button #########################
        self.load_fft_btn = ttk.Button(self.control_frame,text="Calculate FFT",command=self._fft_image) #create a hello button in the control frame
        self.load_fft_btn.pack(pady=40) 

        # FFT Brightness slide bar #
        self.fft_bright_slide_frame = ttk.Frame(self.control_frame,padding=2)
        self.fft_bright_slide_frame.pack(side = 'top',pady=5,fill='x')

        self.fft_brightslide_label = ttk.Label(self.fft_bright_slide_frame,text='Brightness')
        self.fft_brightslide_label.pack(side='left',expand=1,fill='x')

        self.fft_brightness_value = tk.DoubleVar(value=50)
        self.fft_bright_slidebar = ttk.Scale(self.fft_bright_slide_frame,from_=0,to=100,variable=self.fft_brightness_value,
                                             command=self._change_fft)
        self.fft_bright_slidebar.pack(side='left')

        self.fft_bright_reset = ttk.Button(self.fft_bright_slide_frame,text='reset',command=self._reset_br_fft)
        self.fft_bright_reset.pack(side='left')


        #  FFT Contrast slide bar  #
        self.fft_contrast_slide_frame = ttk.Frame(self.control_frame,padding=2)
        self.fft_contrast_slide_frame.pack(side = 'top',pady=5,fill='x')

        self.fft_contrastslide_label = ttk.Label(self.fft_contrast_slide_frame,text='Contrast')
        self.fft_contrastslide_label.pack(side='left',expand=1,fill='x')

        self.fft_contrast_value = tk.DoubleVar(value=50)
        self.fft_contrast_slidebar = ttk.Scale(self.fft_contrast_slide_frame,from_=0,to=100,variable=self.fft_contrast_value,
                                               command=self._change_fft)
        self.fft_contrast_slidebar.pack(side='left')

        self.fft_contrast_reset = ttk.Button(self.fft_contrast_slide_frame,text='reset',command=self._reset_ct_fft)
        self.fft_contrast_reset.pack(side='left')


        # FFT gamma slide bar #
        self.fft_gamma_slide_frame = ttk.Frame(self.control_frame,padding=2)
        self.fft_gamma_slide_frame.pack(side = 'top',pady=5,fill='x')

        self.fft_gammaslide_label = ttk.Label(self.fft_gamma_slide_frame,text='Alpha')
        self.fft_gammaslide_label.pack(side='left',expand=1,fill='x')

        self.fft_gamma_value = tk.DoubleVar(value=50)
        self.fft_gamma_slidebar = ttk.Scale(self.fft_gamma_slide_frame,from_=0,to=100,variable=self.fft_gamma_value,
                                            command=self._change_fft)
        self.fft_gamma_slidebar.pack(side='left')

        self.fft_gamma_reset = ttk.Button(self.fft_gamma_slide_frame,text='reset',command=self._reset_ga_fft)
        self.fft_gamma_reset.pack(side='left')


        ################ Image and FFT Panel ################
        self.image_fft_label = ttk.Label(self.image_fft_frame,text="Image and FFT") #create a hello button in the control frame
        self.image_fft_label.pack() # how the button is packed in the control_frame

  
        # Image #

        # Label 
        self.image_label_frame = ttk.Frame(self.image_fft_frame,padding=2,relief='ridge') # image frame
        self.image_label_frame.pack(side='top',fill='none',expand=0,pady=10) 

        self.image_label = ttk.Label(self.image_label_frame,text="Image") # build and pack label into image frame        
        self.image_label.pack(side='left')

        self.image_reset_btn = ttk.Button(self.image_label_frame,text="Reset View",command=self._reset_img_view) # reset view button
        self.image_reset_btn.pack(side='left',padx=5)

        self.image_cmap_dropdown = ttk.Combobox(self.image_label_frame,values=["Gray","Jet","Hot","Cool"],state='readonly') # colormap dropdown
        self.image_cmap_dropdown.current(0) # set default value
        self.image_cmap_dropdown.bind("<<ComboboxSelected>>",func=self._change_image) # each time select from the drop down, call the _cmap_change function
        self.image_cmap_dropdown.pack(side='left',padx=5)

        self.image_display_frame = ttk.Frame(self.image_fft_frame,padding=2,relief='ridge')
        self.image_display_frame.pack(pady=0,expand=1,fill='both') 

        self.image_canvas = tk.Canvas(self.image_display_frame,highlightthickness=0,width=400,height=400) # suppose the image aspect ratio is 1:1
        self.image_canvas.pack(expand=1)
        self.image_canvas.bind("<MouseWheel>", func=self._on_scroll) # bind mouse wheel event with canvas, whenever mouse wheel event happens, func is called

        # FFT #
        
        # Label frame
        self.fft_label_frame = ttk.Frame(self.image_fft_frame,padding=2) 
        self.fft_label_frame.pack(side='top',fill='none',expand=0,pady=0)

        # Label
        self.fft_label = ttk.Label(self.fft_label_frame,text="FFT") 
        self.fft_label.pack(side = 'left',padx=5)

        # Reset view button
        self.fft_reset_bth = ttk.Button(self.fft_label_frame,text="Reset View",command=self._reset_view_fft) # reset view button
        self.fft_reset_bth.pack(side='left',padx=5)

        # colormap drop down
        self.fft_cmap_dropdown = ttk.Combobox(self.fft_label_frame,values=["Gray","Jet","Hot","Cool"],state='readonly') # colormap dropdown
        self.fft_cmap_dropdown.current(0) # set default value
        self.fft_cmap_dropdown.bind("<<ComboboxSelected>>",func=self._change_fft) # each time select from the drop down, call the _cmap_change function
        self.fft_cmap_dropdown.pack(side='left',padx=5)

        # Canvas: first a frame then a canvas inside the frame
        self.fft_display_frame = ttk.Frame(self.image_fft_frame,relief='ridge',padding=2)
        self.fft_display_frame.pack(side='top',pady=0,expand=1,fill='both') 

        self.fft_canvas = tk.Canvas(self.fft_display_frame,highlightthickness=0,width=400,height=400) # suppose the fft aspect ratio is 1:1
        self.fft_canvas.pack(side='top',expand=1)

        self.fft_canvas.bind("<MouseWheel>", func=self._on_scroll_fft) #  Bind mouse scroll event with self._on_scroll_fft, whenever mouse wheel event happens, func is called
        self.fft_canvas.bind("<Button-1>",func=self._on_press_fft)
        self.fft_canvas.bind("<B1-Motion>",func = self._on_drag_fft)
        self.fft_canvas.bind("<ButtonRelease-1>",func = self._on_release_fft)


        # show distance from FFT
        self.fft_dist_frame = ttk.Frame(self.image_fft_frame,padding=2)
        self.fft_dist_frame.pack(side='top')

        self.fft_dist_label = ttk.Label(self.fft_dist_frame,text = 'Distance in real space:')
        self.fft_dist_label.pack(side='left')


        self.fft_dist_var  =tk.StringVar()
        self.fft_dist = ttk.Entry(self.fft_dist_frame,textvariable=self.fft_dist_var,width=10)
        self.fft_dist.pack(side='left')


        self.fft_dist_unit_label = ttk.Label(self.fft_dist_frame,text="Å")
        self.fft_dist_unit_label.pack(side='left')

######### functions for widgets #########

######################## Image panel ############################

    def _on_press_fft(self,press):

        print('create line on FFT')

        self._start = press.x,press.y

        self.line_id = self.fft_canvas.create_line(press.x,press.y,press.x,press.y,
                                                   fill='red',width=2)
        
    def _on_drag_fft(self,e):
        x0,y0 = self._start
        self.fft_canvas.coords(self.line_id,x0,y0,e.x,e.y)

    def _on_release_fft(self,e):
        self._end = e.x,e.y
        self._cal_dist()
        x0,y0 = self._start
        xe,ye = self._end

        ix0 = (x0-self.offx)/self.scale
        iy0 = (y0-self.offy)/self.scale
        ixe = (xe-self.offx)/self.scale
        iye = (ye-self.offy)/self.scale
        self.fft_line_coor = (ix0,iy0,ixe,iye)

    def _cal_dist(self):

        # canvas coordinates
        x0,y0 = self._start
        xe,ye = self._end

        # convert to image cooridnates
        ix0 = (x0-self.offx)/self.scale
        iy0 = (y0-self.offy)/self.scale

        ixe = (xe-self.offx)/self.scale
        iye = (ye-self.offy)/self.scale

        # Distance in image pixel
        dist_px = np.sqrt((ixe-ix0)**2+(iye-iy0)**2)
        self.dist_from_fft = round(1/(dist_px*self.rpx),2)
        self.fft_dist_var.set(str(self.dist_from_fft))
        

    def _show_fft_dist(self):
        if self.dist_from_fft is None:
            self.fft_dist_var.set('')
        else:
            self.fft_dist_var.set(str(self.dist_from_fft))

    def _change_image(self,val=None):

        # show image at selected brightness, contrast and alpha
        # apply b, c, g and rendering
        # b, c, g and cmap change in parallel
        # idea is to show image at current b, c, g, cmap, self.scale, self.offx, self.offy
        if self.img_pil is None:
            return
        
        b = float(self.img_brightness_value.get())-50 # value from slide bar, 75,50
        c = float(self.img_contrast_value.get())/50.0
        g = float(self.img_gamma_value.get())/50.0

        base = self.gray_array.astype(np.float32)
        mu = 128.0

        # apply contrast and brighentss
        img = (base-mu)*c + mu + b
        img = np.clip(img,0,255)

        # apply gamma
        img = (img/255.0)**(1/max(1e-6,g))*255.0
        img = np.clip(img,0,255).astype(np.uint8)

        # set the colormap
        cmap = self.image_cmap_dropdown.get()

        if cmap == 'Gray':
            img
        elif cmap =='Jet':
            img = cv2.applyColorMap(img,cv2.COLORMAP_JET)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif cmap =='Hot':
            img = cv2.applyColorMap(img,cv2.COLORMAP_HOT)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif cmap =='Cool':
            img = cv2.applyColorMap(img,cv2.COLORMAP_COOL)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        self.img_pil = Image.fromarray(img) #original size
        self._render_image()

    def _reset_br_img(self):
        self.img_brightness_value.set(50)
        self._change_image()

    def _reset_ct_img(self):
        self.img_contrast_value.set(50)
        self._change_image()

    def _reset_ga_img(self):
        self.img_gamma_value.set(50)
        self._change_image()
            
    def _fit_to_canvas_image(self):

        # calculate the scale and offset for rendering 
        # fit image to canvas size
        # used for resetting view

        if self.img_pil is None:
            return
        
        cw = self.image_canvas.winfo_width()
        ch = self.image_canvas.winfo_height()
        w0,h0 = self.img_pil.size

        s = min(cw/w0,ch/h0)
        final_scale = max(min(s,self.max_scale),self.min_scale)

        self.scale = final_scale # scale to fit image to the canvas
        self.offx = (cw - w0*self.scale)//2
        self.offy = (ch - h0*self.scale)//2

    def _render_image(self):

    # render image on the canvas at current self.scale, self.offx, self.offy 

        if self.img_pil is None:
            return
        
        # original image size
        w0,h0 = self.img_pil.size

        # image dimension after scaling
        nw = max(1,int(round(w0*self.scale)))
        nh = max(1,int(round(h0*self.scale)))

        disp = self.img_pil.resize((nw,nh),Image.Resampling.LANCZOS) # resize the image for display

        self.tk_img = ImageTk.PhotoImage(disp) # convert the PIL image to Tkinter image object
        self.image_canvas.delete("all") # clear the canvas

        self.image_canvas.create_image(self.offx,self.offy,image=self.tk_img,anchor='nw') # render image from the top-left anchor point

    def _fit_to_canvas_fft(self):

        # calculate the scale and offset for rendering 
        # fit image to canvas size
        # used for resetting view

        if self.fft_pil is None:
            messagebox.showerror("Error","FFT not calculated")
        
        cw = self.fft_canvas.winfo_width()
        ch = self.fft_canvas.winfo_height()
        w0,h0 = self.fft_pil.size

        s = min(cw/w0,ch/h0)
        final_scale = max(min(s,self.max_scale),self.min_scale)

        self.scale = final_scale # scale to fit image to the canvas
        self.offx = (cw - w0*self.scale)//2
        self.offy = (ch - h0*self.scale)//2

    def _on_scroll(self,event):
        # event is the mouse wheel event
        # event.delta: scroll value, scoll up: positive, scroll down: negative
        # event.x, event.y: mouse position relative to the canvas
        # zoom and render at mouse position
        # each scroll step, self._zoom_at is called once
        if event.delta > 0: #scroll up
            self._zoom_at(1.1,event.x,event.y)

        if event.delta < 0: # scroll down
            self._zoom_at(1/1.1,event.x,event.y)

    def _zoom_at(self,factor,cx,cy):
        # zoom the image at mouse position (cx,cy) by factor
        # set the new scale and offset, depending on the scroll event and mouse position
        # factor: scroll zoom factor
        # (cx,cy): mouse position in canvas coordinates
        if self.img_pil is None:
            return
        
        old = self.scale
        new = old * factor 
        # initially self.scale is the scale to fit image to canvas, now zooming determined by scrolling
        scroll_scale = max(self.min_scale,min(self.max_scale,new))

        # position in image coordinates
        ix = (cx-self.offx)/old
        iy = (cy-self.offy)/old

        # set new scale, 
        self.offx = cx-ix*scroll_scale
        self.offy = cy-iy*scroll_scale
        self.scale = scroll_scale

        self._render_image()

    def _reset_img_view(self):
        # reset the image view to fit to canvas
        self._fit_to_canvas_image()
        self._render_image()


######################## FFT panel ######################
    def _render_fft(self):

        # original image size
        w0,h0 = self.fft_pil.size

        # image dimension after scaling
        nw = max(1,int(round(w0*self.scale)))
        nh = max(1,int(round(h0*self.scale)))

        disp = self.fft_pil.resize((nw,nh),Image.Resampling.LANCZOS) # resize the image for display

        self.tk_fft = ImageTk.PhotoImage(disp) # convert the PIL image to Tkinter image object
        self.fft_canvas.delete("all") # clear the canvas
        self.fft_canvas.create_image(self.offx,self.offy,image=self.tk_fft,anchor='nw') # render image from the top-left anchor point

        if self.fft_line_coor:
            ix0,iy0,ixe,iye = self.fft_line_coor
            self.fft_canvas.create_line(ix0*self.scale+self.offx,
                                   iy0*self.scale+self.offy,
                                   ixe*self.scale+self.offx,
                                   iye*self.scale+self.offy,
                                   fill='red',width=2)
    
    def _fft_image(self):

        if self.gray_array is None:
            messagebox.showerror("Error","No image loaded")
            return
        
         # compute FFT
         # self.gray_array is the grayscale numpy array of the image, original size
         # compute 2D FFT and shift the zero frequency component to the center
         # compute log scale spectrum for better visualization

        img_array = self.gray_array
        f = np.fft.fft2(img_array)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        spectrum = np.log1p(mag)

        # normalize to uint8

        s_min,s_max = np.min(spectrum), np.max(spectrum)
        spec_u8 = ((spectrum-s_min)/(s_max-s_min)*255).astype(np.uint8)

        self.gray_fft = spec_u8
        self.fft_pil = Image.fromarray(spec_u8)

        self._fit_to_canvas_fft()
        self.fft_canvas.delete("all")
        self._render_fft()
  
    def _change_fft(self,value=None):

        # show image at selected brightness, contrast and alpha
        # apply b, c, g and rendering
        if self.fft_pil is None:
            return
        
        b = float(self.fft_brightness_value.get())-50 # value from slide bar, 75,50
        c = float(self.fft_contrast_value.get())/50.0
        g = float(self.fft_gamma_value.get())/50.0

        base = self.gray_fft.astype(np.float32)
        mu = np.mean(base)

        # apply contrast and brighentss
        img = (base-mu)*c + mu + b
        img = np.clip(img,0,255)

        # apply gamma
        img = (img/255.0)**(1/max(1e-6,g))*255.0
        img = np.clip(img,0,255).astype(np.uint8)

        # set the colormap
        cmap = self.fft_cmap_dropdown.get()


        if cmap == 'Gray':
            img
        elif cmap =='Jet':
            img = cv2.applyColorMap(img,cv2.COLORMAP_JET)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif cmap =='Hot':
            img = cv2.applyColorMap(img,cv2.COLORMAP_HOT)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif cmap =='Cool':
            img = cv2.applyColorMap(img,cv2.COLORMAP_COOL)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        self.fft_pil = Image.fromarray(img) #original size
        self._render_fft()

    def _reset_br_fft(self):
        self.fft_brightness_value.set(50)
        self._change_fft()

    def _reset_ct_fft(self):
        self.fft_contrast_value.set(50)
        self._change_fft()

    def _reset_ga_fft(self):
        self.fft_gamma_value.set(50)
        self._change_fft()

    def _reset_view_fft(self):
        self._fit_to_canvas_fft()
        self._render_fft()

    def _on_scroll_fft(self,event):
        # event is the mouse wheel event
        # event.delta: scroll value, scoll up: positive, scroll down: negative
        # event.x, event.y: mouse position relative to the canvas
        # zoom and render at mouse position
        # each scroll step, self._zoom_at is called once
        if event.delta > 0: #scroll up
            self._zoom_at_fft(1.1,event.x,event.y)

        if event.delta < 0: # scroll down
            self._zoom_at_fft(1/1.1,event.x,event.y)

    def _zoom_at_fft(self,factor,cx,cy):
        # zoom the image at mouse position (cx,cy) by factor
        # set the new scale and offset, depending on the scroll event and mouse position
        # factor: scroll zoom factor
        # (cx,cy): mouse position in canvas coordinates
        if self.fft_pil is None:
            return
     
        old = self.scale
        new = old * factor 
        # initially self.scale is the scale to fit image to canvas, now zooming determined by scrolling
        scroll_scale = max(self.min_scale,min(self.max_scale,new))

        # position in image coordinates
        ix = (cx-self.offx)/old
        iy = (cy-self.offy)/old

        # set new scale, 
        self.offx = cx-ix*scroll_scale
        self.offy = cy-iy*scroll_scale
        self.scale = scroll_scale
        self._render_fft()

    def _load_display_image(self):

        if self.px_input_box.get()=='':
            messagebox.showerror('Error', 'Please fill in the pixel size')
            return
        file = filedialog.askopenfilename(title="Select an Image File",filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),("All Files", "*.*")])
        if file:
            print(f"Selected file: {file}")
            image = cv2.imread(file, cv2.IMREAD_UNCHANGED) # load the selected image, numpy array in BGR format
            if image is None:
                messagebox.showerror("Error", "Failed to load image.")
                return
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert BGR np array to RGB np array
            self.gray_array = image_rgb.copy() 
            self.img_pil = Image.fromarray(image_rgb) # convert numpy array to PIL image object, grayscale
            w,h = self.img_pil.size
            self.px = self.px_input_box.get()
            self.rpx = 1/(w*float(self.px_input_box.get()))
            self.image_canvas.delete("all")
            self.fft_canvas.delete("all")

            # first fir the image to the canvas and then render at chosen scale
            self._fit_to_canvas_image()
            self._render_image()


#%%

if __name__ == "__main__":
    root = tk.Tk()  # create a tinker root window
    app = App(root) # intialize, define GUI interface and functions
    root.mainloop() # switch to the GUI loop, user interactions from here

    print("Exit GUI")

