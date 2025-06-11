import os
import io
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from threading import Thread
from time import sleep
from label_manager import LabelManager
from image_manager import ImageManager
from rev_vision_v2 import RevVision
import time

class RevGUI:
    def __init__(self, master):
        self.master = master
        self.master.title('REV Vision')

        # Create a frame for the GUI and center it
        self.frame = tk.Frame(self.master)
        self.frame.pack(expand=True, padx=10, pady=10)
        self.frame.grid_rowconfigure(0,  weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # Create a border for the Controls
        self.border = tk.Frame(self.frame, borderwidth=2, relief="groove")
        self.border.grid(column=0, row=0, sticky="nsew")

        # Create a border for the input ImageFrame
        self.image_border = tk.Frame(self.frame, borderwidth=2, relief="groove")
        self.image_border.grid(column=1, row=0, sticky="nsew")

        # Create a label to display the input image
        self.input_frame = tk.Label(self.image_border)
        self.input_frame.grid(column = 1, row = 1, sticky="nsew", padx = 10, pady = 10)
        self.input_frame.bind("<ButtonRelease-1>", self.click_event)


        # Create a label to display the output image
        self.output_frame = tk.Label(self.image_border)
        self.output_frame.grid(column = 1, row = 2, sticky="nsew", padx = 10, pady = 10)

        # Create a "Load Image Directory" button
        self.load_button = tk.Button(self.border, text="Load Directory", command=self.load_directory)
        self.load_button.grid(column = 1, row = 1, padx = 10, pady = 10, sticky="w")

        # Create a Label for channel OptionMenu
        self.filter_label = tk.Label(self.border, text=' Filter Object ', relief=tk.FLAT)
        self.filter_label.grid(column = 1, row = 2, padx = 10, pady = 10, sticky="sw")

        # Create a Option menu for setting channel
        self.filter_var = tk.StringVar()
        self.filter_var.set("None")
        self.filter_option = tk.OptionMenu(self.border, self.filter_var, "None", command=self.filter_select)
        self.filter_option.config(width=13)
        self.filter_option.grid(column = 1, row = 3, padx = 10, pady = 10, sticky="nsew")

        # Create a Label for channel OptionMenu
        self.capture_label = tk.Label(self.border, text=' Capture Object ', relief=tk.FLAT)
        self.capture_label.grid(column = 1, row = 4, padx = 10, pady = 10, sticky="sw")

        # Create a Option menu for setting channel
        self.capture_var = tk.BooleanVar()
        self.capture_var.set(False)
        # self.capture_option = tk.OptionMenu(self.border, self.capture_var, "None", command=self.capture_select)
        # self.capture_option.config(width=13)
        # self.capture_option.grid(column = 1, row = 5, padx = 10, pady = 10, sticky="nsew")
        self.caputure_button = tk.Checkbutton(self.border, text="CAPTURE", variable=self.capture_var, onvalue=True, offvalue=False, command=self.capture_select)
        self.caputure_button.grid(column = 1, row = 5, padx = 10, pady = 10, sticky="nsew")

        ##
        ## Variables
        self.file_path = ''
        self.img_manager = None
        self.updateFrameId = 0
        self.player_state = False
        self.rev_vision = None
        self.player_state = False
        self.label_manager = LabelManager()
        self.current_label = 0
        self.pred_labels = []
        self.pred_class = []
        self.cur_filter = 'None'
        self.cur_capture = False


    def load_directory(self):
        # Open a file selection dialog box to choose an image file
        self.file_path = filedialog.askdirectory(title="Select Input Folder")
        self.img_manager = ImageManager(self.file_path)
        self.rev_vision = RevVision()
        self.frame_num = self.img_manager.get_num()
        self.showFrameController()
        self.showImage(0)
        # print(self.file_path)
        self.img_manager.clear_capture()

        
    def showFrameController(self):
        # Create a "play" button
        self.set_label_button = tk.Button(self.image_border, text="play", command=self.play_frame)
        self.set_label_button.grid(column = 1, row = 3, padx = 10, pady = 10, sticky="nw")

        # Create a "pause" button
        self.set_label_button = tk.Button(self.image_border, text="pause", command=self.pause_frame)
        self.set_label_button.grid(column = 1, row = 3, padx = 70, pady = 10, sticky="nw")

        #  Create a Scale widget for setting Slice ID        
        self.slice_var = tk.IntVar()
        self.slice_var.set(0)
        self.slice_scale = tk.Scale(self.image_border, width=20, length = self.img_manager.width, 
                                    from_=0, to=self.frame_num - 1, orient=tk.HORIZONTAL, label="Frame ID", 
                                    variable=self.slice_var, command=self.update_frameId)
        self.slice_scale.grid(column = 1, row = 4, sticky="sw", padx = 10, pady = 10)
        #self.slice_scale.bind("<ButtonRelease-1>", self.updateFrameId)

    def click_event(self, event):
        evt_name = str(event)
        print(evt_name)

    def update_frameId(self, event):
        evt_name = str(event)
        print(evt_name)
        frame_id = self.slice_var.get()
        print('frame_id: ', frame_id)
        self.showImage(frame_id)

    def filter_select(self, event):
        evt_name = str(event)
        print('filter_select', evt_name)
        if (evt_name != self.cur_filter):
            if evt_name == 'None':
                self.current_label = 0
            else:
                self.current_label = self.label_manager.get_label(evt_name)
            frame_id = self.slice_var.get()
            self.showImage(frame_id)
            self.cur_filter = evt_name

    def capture_select(self):
        capture_on = self.capture_var.get()
        print('capture_select: ', capture_on)
        if (capture_on != self.cur_capture):
            frame_id = self.slice_var.get()
            self.cur_capture = capture_on
            self.showImage(frame_id)


    def play_frame(self):
        player_thread = Thread(target=self.start_playback)
        self.player_state = True
        player_thread.start()
        
    def pause_frame(self):
        self.player_state = False

    def start_playback(self):
        frame_id = self.slice_var.get()
        print('start_playback', frame_id)
        for i in range(frame_id, self.frame_num):
            # print(self.player_state)
            if self.player_state == False:
                break
            start_time = time.perf_counter()
            self.showImage(i)
            end_time = time.perf_counter()
            self.slice_var.set(i)
            processing_time = end_time - start_time
            rest_time = 0.066 - processing_time
            # print(rest_time)
            if (rest_time > 0):
                sleep(0.066)

    def showImage(self, frame_id, update_ui = True):
        in_img, out_img = self.predict_objects(frame_id, update_ui, self.current_label, self.cur_capture)
        in_photo = self.img_manager.convert_photo(in_img)
        out_photo = self.img_manager.convert_photo(out_img)
        
        self.input_frame.configure(image=in_photo)
        self.input_frame.image = in_photo

        self.output_frame.configure(image=out_photo)
        self.output_frame.image = out_photo

    def predict_objects(self, frame_id, update_ui, filter, capture):
        img = self.img_manager.get_img(frame_id)
        in_img, out_img, pred_cls, labels = self.rev_vision.segment_instance(img, frame_id=frame_id, label_filter= filter, capture = capture)
        self.pred_labels = labels
        self.pred_class = pred_cls
        if update_ui:
            self.update_ui(labels)

        return in_img, out_img

    def update_ui(self, labels):
        unique_labels = list(set(labels))
        new_options = [self.label_manager.get_name(i - 1) for i in unique_labels]
        print('new options: ', new_options)
        new_options.insert(0, 'None')
        # menu_capture = self.capture_option["menu"]
        # menu_capture.delete(0, "end") 
        menu_filter = self.filter_option['menu']
        menu_filter.delete(0, "end")
        for option in new_options:
            # command_func = lambda val=option: (self.capture_var.set(val), self.capture_select(val))
            # menu_capture.add_command(label=option, command=command_func)
            command_func = lambda val=option: (self.filter_var.set(val), self.filter_select(val))
            menu_filter.add_command(label=option, command=command_func)
            # menu_capture.add_command(label=option, command=lambda value=option: self.capture_var.set(value))
            # menu_capture.add_command(label=option, command=lambda value=option: self.capture_select(value))
            # menu_filter.add_command(label=option, command=lambda value=option: self.filter_var.set(value))
            # menu_filter.add_command(label=option, command=lambda value=option: self.filter_select(value))
            #menu.add_command(label=option, command=lambda value=option: self.capture_select(value)) # Also call your command


        
if __name__ == "__main__":
    root = tk.Tk()
    gui = RevGUI(root)
    root.mainloop()