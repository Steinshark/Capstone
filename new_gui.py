
import tkinter                                      # main library for handling GUI
import threading                                    # main library for threading
import tkinter.scrolledtext                         # allows for text display in GUI
import time


from tkinter    import BitmapImage, Menu, Frame, TOP, RAISED
from GuiTools   import *
from Utilities  import *
from modes      import *

# A clean container for imported files
class ImportedFile:
    def __init__(self,filepath,contents_as_rb):
        self.filepath = filepath
        self.contents_as_rb = contents_as_rb

    def __repr__(self):
        return 'new_imported_file\n\t' + str(self.filepath) + '\n\t' + str(self.contents_as_rb) + '\n'

class GUI_APP:
    def __init__(self,width,height,w_name):
        #set settings
        self.initialize_settings(width,height,w_name)

        #create a window
        self.initialize_window()

        #create options menu
        self.change_mode('worksession',4)

        #create frames to organize layout

        #create interactables (i.e. buttons)

        self.timer = time.time()
        self.draw()
        self.window.mainloop()
        while self.settings['running']:
            try:
                self.update_window_tasks()
                self.window.update_idletasks()
            except tkinter.TclError:
                print("Shutting down, goodbye!")
                return

    def initialize_window(self):
        self.window = tkinter.Tk()
        self.window.geometry(get_window_size_as_text(self))
        self.window.title(get_window_title_as_text(self))
        #h = tkinter.PhotoImage(os.getcwd()+'/logo.ico')
        #self.window.wm_iconphoto(True,h)

        # Update some settings that we get only after we instantiate
        # the window
        self.settings['screen_res_width'    ]   = self.window.winfo_screenwidth()
        self.settings['screen_res_height'   ]   = self.window.winfo_screenheight()
        self.settings['current_width'       ]   = self.window.winfo_width()
        self.settings['current_height'      ]   = self.window.winfo_height()
    # Change the layout of the GUI window
    def change_mode(self,mode,windows):
        self.mode = self.modes[mode](self,n_frames = windows)

    # Window settings and data (files, etc...) initialized here
    def initialize_settings(self,w,h,n):
        self.settings   =   {\
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
        self.data       =   {\
                            'loaded_files'          :   dict(),\
                            'models'                :   dict(),\
                        }
        self.tools      =   {\
                            'help'                  :   dict()
                        }
        self.frames     =   {

                        }
        self.modes      =   {\
                            'splash'                        :   Splash,\
                            'worksession'                   :   WorkSession
                        }
        self.viewports  =   {\

        }

    def draw(self):
        self.window.grid()


    # Wrapper for the update method and
    # allows for additional tasks to be
    # completed on every redraw
    def update_window_tasks(self):

        self.draw()



if __name__ == '__main__':
    GUI_APP(800,600,'SEAL Screener')
