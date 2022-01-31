# Needed for the GUI_APP
import tkinter                                      # main library for handling GUI
import threading                                    # main library for threading
import tkinter.scrolledtext                         # allows for text display in GUI
import time
# Needed for Utilities
from tkinter import BitmapImage, Menu, Frame, TOP
from Utilities import *

# A clean container for imported files
class ImportedFile:
    def __init__(self,filepath,contents_as_rb):
        self.filepath = filepath
        self.contents_as_rb = contents_as_rb

    def __repr__(self):
        return 'new_imported_file\n\t' + str(self.filepath) + '\n\t' + str(self.contents_as_rb) + '\n'

class GUI_APP:
    def __init__(self,width,height,w_name):
        self.sessions = {}
        #set settings
        self.initialize_settings(width,height,w_name)

        #create a window
        self.initialize_window()

        #create options menu
        self.initialize_options_menu()

        #create frames to organize layout
        self.initialize_frames()

        #create interactables (i.e. buttons)
        self.initialize_interactables()
        self.timer = time.time()
        while self.settings['running']:
            try:
                self.update_window_tasks()
                self.window.update_idletasks()
            except tkinter.TclError:
                print("Shutting down, goodbye!")
                return


    def initialize_window(self):
        self.window = tkinter.Tk()
        self.window.geometry(Utilities.get_window_size_as_text(self))
        self.window.title(str(self.settings['window_name']))
        #h = tkinter.PhotoImage(os.getcwd()+'/logo.ico')
        #self.window.wm_iconphoto(True,h)

        # Update some settings that we get only after we instantiate
        # the window
        self.settings['screen_res_width']   = self.window.winfo_screenwidth()
        self.settings['screen_res_height']  = self.window.winfo_screenheight()
        self.settings['current_width']      = self.window.winfo_width()
        self.settings['current_height']     = self.window.winfo_height()

    def initialize_options_menu(self):

        # Initialize a standard TKINTER menu
        self.options_menu = tkinter.Menu(self.window)
        self.menuFrame = tkinter.Frame(self.window)
        self.window.config(menu=self.options_menu)

        fileMenu = Menu(self.options_menu)

        subset = Menu(fileMenu)
        subset.add_command(label='Exit',command=self.window.quit)
        fileMenu.add_cascade(label='Import',menu=subset,underline=0)



    # Window settings and data (files, etc...) initialized here
    def initialize_settings(self,w,h,n):
        self.settings = {\
                            'init_width'                    :   w,\
                            'init_height'                   :   h,\
                            'current_width'                 :   w,\
                            'current_height'                :   h,\
                            'screen_res_width'              :   0,\
                            'screen_res_height'             :   0,\
                            'window_name'                   :   n,\
                            'resize'                        :   False,\
                            'running'                       :   True,\
                            'main color'                    : '#2E3338'
                        }
        self.data =     {\
                            'loaded_files'          :   dict(),\
                            'models'                :   dict(),\
                        }

        self.tools =    {\
                            'help'                  :   dict()
                        }

    # Frames are the basic way of organizing the layout of the TKinter window
    def initialize_frames(self):
        self.frames = {
                                    'splash'                    :   None

        }
        self.frames['splash'] = {
                                    'toolbar'                   :   Frame(self.window),\
                                    'main'                      :   Frame(self.window)
        }


        self.frames['splash']['toolbar'].pack(side=TOP)

    # Creates the buttons that will be used to import files, export, etc...
    # Each interactable is (currently) paired with a label
    def initialize_interactables(self):
        '''
        self.labels  = {\
                            'import dataset'        :   (tkinter.Label(master = self.frames['IO'][0],text = 'Import a file', pady=5,padx=5)),\
                            'import datapoint'      :   (tkinter.Label(master = self.frames['IO'][0],text = 'Import files', pady=5,padx=5)),\
                            'export result'         :   (tkinter.Label(master = self.frames['IO'][0],text = 'Export a file', pady=5,padx=5)),\
                            'save session'          :   (tkinter.Label(master = self.frames['IO'][0],text = 'Save Session' , pady=5,padx=5)),\
                            'load session'          :   (tkinter.Label(master = self.frames['IO'][0],text = 'Load Session' , pady=5,padx=5))\

                       }

        self.entries = {\
                            'import'                :   ((tkinter))\
                       }

        self.buttons = {\
                            'import dataset'        :   (tkinter.Button(master = self.frames['IO'][0],width=20,text='import dataset',command=lambda : Utilities.import_files(self))),\
                            'import datapoint'      :   (tkinter.Button(master = self.frames['IO'][0],width=20,text='import datapoint',command=lambda : Utilities.import_file(self))),\
                            'export result'         :   (tkinter.Button(master = self.frames['IO'][0],width=20,text='export',command=lambda : Utilities.export_file(self))),\
                            'save session'          :   (tkinter.Button(master = self.frames['IO'][0],width=20,text='save session',command=lambda : Utilities.save_session(self,True))),\
                            'load session'          :   (tkinter.Button(master = self.frames['IO'][0],width=20,text='load session',command=lambda : Utilities.load_session(self))),\
                       }
        '''
        self.live_text = tkinter.scrolledtext.ScrolledText(master=self.frames['splash']['main'],width=50,height=20)

    def draw(self):
        # Main spans the entirety of the top
        '''
        frame, fill = self.frames['title']
        frame.pack(fill=fill)

        label = tkinter.Label(frame, text="NLP SEAL Screener",bg = self.settings['main color'],pady=5,padx=5)
        label.grid(row=0,column=0)

        frame,fill = self.frames['IO']
        frame.pack(fill=fill)
        '''
        '''
        for i in range(len(list(self.labels.keys()))):
            label = list(self.labels.values())[i]
            button = list(self.buttons.values())[i]
            label.grid(row=i+1,column = 0,stick='W')
            button.grid(row=i+1,column = 1)
        '''

        self.live_text.pack()



    # Wrapper for the update method and
    # allows for additional tasks to be
    # completed on every redraw
    def update_window_tasks(self):
        '''
        # "super"
        if time.time() - self.timer > 2:
            Utilities.save_session(self,False)
            self.timer = time.time()
        self.window.update()
        # rewraw if the window is resized
        if not self.window.winfo_width() == self.settings['current_width'] or not self.window.winfo_height() == self.settings['current_height']:
            self.settings['current_width']      = self.window.winfo_width()
            self.settings['current_height']     = self.window.winfo_height()
        '''
        self.draw()



if __name__ == '__main__':
    GUI_APP(800,600,'SEAL Screener')
