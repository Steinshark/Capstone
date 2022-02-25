
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
        import sys
        self.change_mode('worksession',int(sys.argv[1]))

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

    # Call all methods to initialize the GUI and 
    # initialze all fields contained within
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
    # contains the init method of the class 'mode'              called with reference to self   and num interveiws to show
        self.mode = self.modes[mode](                           self,                           n_frames = windows)


    # All GUI fields are init'ed here. a GUI will have a:
        # settings 
        # data
        # tools 
        # frames
        # modes 
        # viewports
    def initialize_settings(self,w,h,n):

        # Used to globally track settings that are pertinent to 
        # the gui itself. height, width, name, color, etc... 
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

        # Used to store the data (interviews and NLP models) that the
        # APP has access to.  
        self.data       =   {\
                            'loaded_files'          :   dict(),\
                            'models'                :   dict(),\
                        }

        # *DEPRECATED* 
        # Used to keep track of tools available to the GUI.
        # CURRENT IMPLEMENTATION DOES NOT RELY ON THIS. this 
        # data is stored in 'Modes.py' 
        self.tools      =   {\
                            'help'                  :   dict()
                        }

        # Used to keep track of the frames (tkinter gui building block)
        # that are currently active in the gui. 'Modes.py' handles the 
        # rearranging of these  
        self.frames     =   {

                        }

        # Handles the switching of the GUI layout. Currently 2 modes exists.
        # The splash screen layout and the worksession layout.
        self.modes      =   {\
                            'splash'                        :   Splash,\
                            'worksession'                   :   WorkSession
                        }

        # A viewport holds one interview editing environment. It includes the interview 
        # display, the tools, the comment bar, etc... 
        self.viewports  =   {\

        }

    # *DEPRECATED*
    # Currently not used.
    def draw(self):
        self.window.grid()



    # *DEPRECATED*
    # Currently not used.
    # Wrapper for the update method and
    # allows for additional tasks to be
    # completed on every redraw
    def update_window_tasks(self):

        self.draw()



if __name__ == '__main__':
    GUI_APP(800,600,'SEAL Screener')
