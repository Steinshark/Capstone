import os                                           # allows access to filepath
from tkinter import Menu, RAISED, Frame
from tkinter.scrolledtext import *
from tkPDFViewer import tkPDFViewer


class Splash:
    def __init__(self, APP_REF):
        self.load_menu(APP_REF)
        self.load_frames(APP_REF)
        self.pack_frames(APP_REF)

    def load_menu(self,APP_REF):
        self.toolbar = Menu(APP_REF.window,relief=RAISED,background='blue',fg='#125361')

        # Init each dropdown that we want
        self.tool_dropdown = {
                                'file'      :   Menu(self.toolbar, tearoff=0),\
                                'edit'      :   Menu(self.toolbar, tearoff=0)
        }

        # For each dropdown, add commands
        self.tool_dropdown['file'].add_command(label='new',command = lambda x : x)
        self.tool_dropdown['file']['bg']= '#125361'


        self.tool_dropdown['edit'].add_command(label='import',command = lambda x : x)
        #self.tool_dropdown['file'].config(bg='red')

        # add the dropdown to the toolbar
        self.toolbar.add_cascade(label='File',menu=self.tool_dropdown['file'])
        self.toolbar.add_cascade(label='edit',menu=self.tool_dropdown['edit'])

        # add the bar to the window
        APP_REF.window.config(menu=self.toolbar)

    def load_frames(self,APP_REF):
        APP_REF.frames = {
                                    'toolbar'                   :   Frame(APP_REF.window,bg='red'),\
                                    'main'                      :   Frame(APP_REF.window)
        }

class WorkSession:
    def __init__(self, APP_REF):
        self.load_menu(APP_REF)
        self.load_frames(APP_REF)

    def load_menu(self,APP_REF):
        self.toolbar = Menu(APP_REF.window,relief=RAISED,background='blue',fg='#125361')

        # Init each dropdown that we want
        self.tool_dropdown = {
                                'file'      :   Menu(self.toolbar, tearoff=0),\
                                'edit'      :   Menu(self.toolbar, tearoff=0)
        }

        # For each dropdown, add commands
        self.tool_dropdown['file'].add_command(label='new',command = lambda x : x)
        self.tool_dropdown['file']['bg']= '#125361'


        self.tool_dropdown['edit'].add_command(label='import',command = lambda x : x)
        #self.tool_dropdown['file'].config(bg='red')

        # add the dropdown to the toolbar
        self.toolbar.add_cascade(label='File',menu=self.tool_dropdown['file'])
        self.toolbar.add_cascade(label='edit',menu=self.tool_dropdown['edit'])

        # add the bar to the window
        APP_REF.window.config(menu=self.toolbar)

    def load_frames(self,APP_REF):
        APP_REF.frames = {
                                    'toolbar'                   :   Frame(APP_REF.window,bg='red'),\
                                    'workspace1'                 :   Frame(APP_REF.window),
                                    'workspace2'                 :   Frame(APP_REF.window)
        }

        #APP_REF.frames['workspace'].columnconfigure(0,weight=1)
        #APP_REF.frames['workspace'].columnconfigure(0,weight=3)


        APP_REF.live_text1 = ScrolledText(APP_REF.window)
        APP_REF.live_text1.grid(column=0,row=0)

        APP_REF.live_text2 = ScrolledText(APP_REF.window)
        APP_REF.live_text2.grid(column=1,row=0)
        #APP_REF.pdf = tkPDFViewer.ShowPdf()
        #APP_REF.pdf2 = APP_REF.pdf.pdf_view(APP_REF.window, pdf_location = r'D:\Website\reciept.pdf')
        #APP_REF.pdf2.grid(column=1,row=0)
