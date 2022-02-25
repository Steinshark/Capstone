import os
from tkinter.filedialog import askopenfiles         # allows for file interaction
from tkinter.filedialog import askopenfile          # allows for file interaction
from tkinter.filedialog import askdirectory         # allows for file interaction
import json




# This class file is the location of all tools that will be called from within the GUI 
# To interact with the GUI instance, name the parameter 'APP_REFERENCE' and pass 'self'
# as the argument when calling from 'gui.py'.

 

# This method is lowkey kinda useless and it just wraps a call to 'print_cwd()'

# parameters: None 
# return:     Filepath
def get_os_root_filepath():
    return os.getcwd()


# This method is used to keep 'gui.py' clean

# parameters: APP_REF
# return:     string of '{width}x{height}'           
def get_window_size_as_text(APP_REFERENCE):
    text = str(APP_REFERENCE.settings['init_width'])
    text += 'x'
    text += str(APP_REFERENCE.settings['init_height'])
    return text


# This method is used to keep 'gui.py' clean, and wraps a getter for the title

# parameters: APP_REF
# return:     string of 'windowname'   
def get_window_title_as_text(APP_REFERENCE):
    return str(APP_REFERENCE.settings['window_name'])


# This method is used to load multiple interview files from disk into memory and place it in the 
# APP_REFERENCE. It loads the filename to a python file object, and then from the python 
# file object to a custom class that holds all pertinent info.  

# parameters: APP_REF
# return:     none   
def import_files(APP_REFERENCE):

    # Choose which filetypes are visible and able to be chosen in the dialog 
    supported_types =   (   ("text files", "*.txt"),\
                            ("word files", "*.docx"),\
                            ("pdf files", "*.pdf"),\
                            ("all files", "*.*")  )# <-Probably will not include in final version


    # Opens blocking tkinter file dialog
    file_list = askopenfiles(mode='rb',filetypes=supported_types)

    # user picks a file(s) which is added to the data dictionary of the APP
    distinct_files = 0

    if not len(file_list) == 0:
        for file in file_list:
            if not file is None and not file.name in APP_REFERENCE.data['loaded_files'].keys():
                distinct_files += 1
                # Add to the running instance's data  dictionary
                APP_REFERENCE.data['loaded_files'][file.name] = ImportedFile(file.name,file.raw.read())

    return


# This method is used to load an interview file from disk into memory and place it in the 
# APP_REFERENCE. It loads the filename to a python file object, and then from the python 
# file object to a custom class that holds all pertinent info.  

# parameters: APP_REF
# return:     none   
def import_file(APP_REFERENCE):

    # Choose which filetypes are visible and able to be chosen in the dialog 
    supported_types =   (   ("text files", "*.txt"),\
                            ("word files", "*.docx"),\
                            ("pdf files", "*.pdf"),\
                            ("all files", "*.*")  )# <-Probably will not include in final version


    file = askopenfile(filetypes=supported_types)

    # user picks a file which is added to the data dictionary of the APP
    if not file is None:
        APP_REFERENCE.data['loaded_files'][file.name] = file
    
    return

class Toolchain:
    pass
