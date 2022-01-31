import os
from tkinter.filedialog import askopenfiles         # allows for file interaction
from tkinter.filedialog import askopenfile          # allows for file interaction
from tkinter.filedialog import askdirectory         # allows for file interaction
from json import loads, dumps



def get_os_root_filepath():
    return os.getcwd()

def get_window_size_as_text(APP_REFERENCE):
    text = str(APP_REFERENCE.settings['init_width'])
    text += 'x'
    text += str(APP_REFERENCE.settings['init_height'])
    return text

def get_window_title_as_text(APP_REFERENCE):
    return str(APP_REFERENCE.settings['window_name'])

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

class Toolchain:
    pass
