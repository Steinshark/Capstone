# This class is meant to make the GUI_APP class cleaner
# "Utilities" will not be instantiated. All methods are
# static
import os                                           # allows access to filepath
import tkinter
from tkinter.filedialog import askopenfiles         # allows for file interaction
from tkinter.filedialog import askopenfile          # allows for file interaction
from tkinter.filedialog import askdirectory         # allows for file interaction
from json import loads, dumps


# A clean container for imported files
class ImportedFile:
    def __init__(self,filepath,contents_as_rb):
        self.filepath = filepath
        self.contents_as_rb = contents_as_rb
        self.contents_as_text = contents_as_rb.decode()

        self.lines = [l for l in self.contents_as_text.split('\n') if not l == '']
        self.words = [w for w in self.contents_as_text.split(' ')  if not w == '']
        self.chars = [c for c in self.contents_as_text  if not c == ' ' or not c =='']

    def __repr__(self):
        return 'new_imported_file\n\t' + str(self.filepath) + '\n\t' + str(self.contents_as_rb) + '\n'


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
                    APP_REFERENCE.live_text.delete('0.0', END)

                    APP_REFERENCE.live_text.insert(tkinter.END,"\tfetched: " + str(file.name.split('/')[-1]) + "\n")
                    APP_REFERENCE.live_text.yview(tkinter.END)

            APP_REFERENCE.live_text.insert(tkinter.END,"Imported " + str(distinct_files) + " new files\n\n")
            APP_REFERENCE.live_text.yview(tkinter.END)

            # Save session upon any new uploads
            Utilities.save_session(APP_REFERENCE,False)
            return


    # Upload single file to app
    @staticmethod
    def import_file(APP_REFERENCE,scrolled_text=None):
        supported_types =   (   ("text files", "*.txt"),\
                                ("word files", "*.docx"),\
                                ("pdf files", "*.pdf"),\
                                # Probably will not include in final version
                                ("all files", "*.*")  )

        file = askopenfile(mode='rb',filetypes=supported_types)

        # user picks a file which is added to the data dictionary of the APP
        if not file is None:
            APP_REFERENCE.data['loaded_files'][file.name] = ImportedFile(file.name,file.raw.read())

        print(f"rb is {APP_REFERENCE.data['loaded_files'][file.name].contents_as_rb}")
        if not scrolled_text is None:
            scrolled_text.delete('0.0',tkinter.END)
            text = APP_REFERENCE.data['loaded_files'][file.name].contents_as_rb.decode()
            scrolled_text.insert(tkinter.END,f"{text}\n")
        else:
            print("oh no")


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
    def import_sessions(APP_REFERENCE):
        sessions = {}
        with open('sessions.txt') as save_states:
            for raw_text in save_states.read():
                return loads(raw_text)



    # Recover the file dictionary to rebuild
    # the most recent file dictionary for the
    # GUI APP. Will always look for 'gui_sess.tmp'
    @staticmethod
    def save_session(APP_REFERENCE):
        sessions = {}
        with open('sessions.txt') as save_states:
            for line in save_states.readlines():
                pass
