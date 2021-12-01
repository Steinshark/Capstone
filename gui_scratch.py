# Needed for the GUI_APP
import tkinter                                      # main library for handling GUI
import threading                                    # main library for threading
import tkinter.scrolledtext                         # allows for text display in GUI
import time
# Needed for Utilities
import os                                           # allows access to filepath
from tkinter.filedialog import askopenfiles         # allows for file interaction
from tkinter.filedialog import askopenfile          # allows for file interaction
from tkinter.filedialog import askdirectory         # allows for file interaction
from tkinter import BitmapImage

# A clean container for imported files
class ImportedFile:
    def __init__(self,filepath,contents_as_rb):
        self.filepath = filepath
        self.contents_as_rb = contents_as_rb

    def __repr__(self):
        return 'new_imported_file\n\t' + str(self.filepath) + '\n\t' + str(self.contents_as_rb) + '\n'

# This class is meant to make the GUI_APP class cleaner
# "Utilities" will not be instantiated. All methods are
# static
class Utilities:

    @staticmethod
    def get_os_root_filepath():
        return os.getcwd()


    @staticmethod
    def get_window_size_as_text(APP_REFERENCE):
        text = str(APP_REFERENCE.settings['init_width'])
        text += 'x'
        text += str(APP_REFERENCE.settings['init_height'])
        return text


    # Upload multiple files to app
    @staticmethod
    def import_files(APP_REFERENCE):

        supported_types =   (   ("text files", "*.txt"),\
                                ("word files", "*.docx"),\
                                ("pdf files", "*.pdf"),\
                                # Probably will not include in final version
                                ("all files", "*.*")  )

        # Opens blocking tkinter file dialog
        file_list = askopenfiles(mode='rb',filetypes=supported_types)
        APP_REFERENCE.live_text.insert(tkinter.END,"Importing Files:\n")
        APP_REFERENCE.live_text.yview(tkinter.END)

        # user picks a file which is added to the data dictionary of the APP
        distinct_files = 0

        if not len(file_list) == 0:
            for file in file_list:
                if not file is None and not file.name in APP_REFERENCE.data['loaded_files'].keys():
                    distinct_files += 1
                    # Add to the running instance's data  dictionary
                    APP_REFERENCE.data['loaded_files'][file.name] = ImportedFile(file.name,file.raw.read())

                    # Print status to the GUI text bar
                    APP_REFERENCE.live_text.insert(tkinter.END,"\tfetched: " + str(file.name.split('/')[-1]) + "\n")
                    APP_REFERENCE.live_text.yview(tkinter.END)

            APP_REFERENCE.live_text.insert(tkinter.END,"Imported " + str(distinct_files) + " new files\n\n")
            APP_REFERENCE.live_text.yview(tkinter.END)

            # Save session upon any new uploads
            Utilities.save_session(APP_REFERENCE,False)
            return


    # Upload single file to app
    @staticmethod
    def import_file(APP_REFERENCE):
        supported_types =   (   ("text files", "*.txt"),\
                                ("word files", "*.docx"),\
                                ("pdf files", "*.pdf"),\
                                # Probably will not include in final version
                                ("all files", "*.*")  )

        file = askopenfile(filetypes=supported_types)

        # user picks a file which is added to the data dictionary of the APP
        if not file is None:
            APP_REFERENCE.data['loaded_files'][file.name] = file
        else:
            #line114
            pass


    # this method will be used to export the
    # results of our NLP magic post-processing
    # of the data
    @staticmethod
    def export_file(APP_REFERENCE):
        # method currently does nothing...
        print("exporting! - (nothing to export yet....)")


    # Save the file dictionary to a file that
    # can be imported at a later time into the
    # GUI APP
    @staticmethod
    def save_session(APP_REFERENCE,verbose):
        file_dict = APP_REFERENCE.data['loaded_files']
        save_file = open("gui_sess.tmp",'w')
        for file in file_dict.values():
            save_file.write(str(file))
        save_file.close()

        if verbose:
            APP_REFERENCE.live_text.insert(tkinter.END,"Saved Session: " + str(len(file_dict)) + " files\n")
            APP_REFERENCE.live_text.yview(tkinter.END)


    # Recover the file dictionary to rebuild
    # the most recent file dictionary for the
    # GUI APP. Will always look for 'gui_sess.tmp'
    @staticmethod
    def load_session(APP_REFERENCE):
        try:
            recovery_file = open("gui_sess.tmp",'r')
        except FileNotFoundError:
            #JENNY, DO A POP-UP HERE!!!
            print("no session file found")

        new_dict = {file_repr.split('\n')[1]: ImportedFile(file_repr.split('\n')[1],file_repr.split('\n')[2]) for file_repr in recovery_file.read().split('new_imported_file')[1:]}
        APP_REFERENCE.live_text.insert(tkinter.END,"Loaded Session: " + str(len(new_dict)) + " files\n")
        APP_REFERENCE.live_text.yview(tkinter.END)
        APP_REFERENCE.data['loaded_files'] = new_dict



class GUI_APP:
    def __init__(self,width,height,w_name):
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
        self.options_menu = tkinter.Menu(self.window,tearoff=0)

        # Add options to menu here ->
        self.options_menu.add_command(label='Exit',command=self.window.quit)
        self.options_menu.add_command(label='Upload File',command=Utilities.import_file)
        #self.options_menu.add_command(label='import interview',command=self.new_method_we_will_build)

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

    # Frames are the basic way of organizing the layout of the TKinter window
    def initialize_frames(self):
        self.frames = {\
                            'title'                  :   (tkinter.Frame(bg=self.settings['main color']),tkinter.X),\
                            'IO'                     :   (tkinter.Frame(),tkinter.X),\
                            'dataset_movement'       :   (tkinter.Frame(),tkinter.X),\
                      }

    # Creates the buttons that will be used to import files, export, etc...
    # Each interactable is (currently) paired with a label
    def initialize_interactables(self):
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
        self.live_text = tkinter.scrolledtext.ScrolledText(master=self.frames['IO'][0],width=50,height=20)
    def draw(self):
        # Main spans the entirety of the top
        frame, fill = self.frames['title']
        frame.pack(fill=fill)

        label = tkinter.Label(frame, text="NLP SEAL Screener",bg = self.settings['main color'],pady=5,padx=5)
        label.grid(row=0,column=0)

        frame,fill = self.frames['IO']
        frame.pack(fill=fill)
        for i in range(len(list(self.labels.keys()))):
            label = list(self.labels.values())[i]
            button = list(self.buttons.values())[i]
            label.grid(row=i+1,column = 0,stick='W')
            button.grid(row=i+1,column = 1)
        self.live_text.grid(row=i+2,column=1)


    # Wrapper for the update method and
    # allows for additional tasks to be
    # completed on every redraw
    def update_window_tasks(self):
        # "super"
        if time.time() - self.timer > 2:
            Utilities.save_session(self,False)
            self.timer = time.time()
        self.window.update()

        # rewraw if the window is resized
        if not self.window.winfo_width() == self.settings['current_width'] or not self.window.winfo_height() == self.settings['current_height']:
            self.settings['current_width']      = self.window.winfo_width()
            self.settings['current_height']     = self.window.winfo_height()
            self.draw()



if __name__ == '__main__':
    GUI_APP(800,600,'SEAL Screener')
