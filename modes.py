import os                                           # allows access to filepath
from tkinter import Menu, RAISED, Frame, Label, Button, Entry, Tk, N, S, E, W, Y, X
from tkinter.scrolledtext import *
from tkPDFViewer import tkPDFViewer
from Utilities import Utilities

class Splash:
    def __init__(self, APP_REF,n_frames=0):
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
    def __init__(self, APP_REF,n_frames=2):

        self.n_frames = n_frames
        APP_REF.window.columnconfigure(0,weight=1)
        APP_REF.window.columnconfigure(1,weight=1)
        APP_REF.window.columnconfigure(2,weight=1)
        APP_REF.window.rowconfigure(0,weight=1)
        APP_REF.window.rowconfigure(1,weight=1)
        APP_REF.window.rowconfigure(2,weight=10)

        APP_REF.window.config(bg='#222226')


        self.load_menu(APP_REF)
        self.load_frames(APP_REF)

    def load_menu(self,APP_REF):
        self.toolbar = Menu(APP_REF.window,relief=RAISED)
        self.toolbar['bg']= '#222222'
        self.toolbar['fg']= '#222222'
        # Init each dropdown that we want
        self.tool_dropdown = {
                                'file'      :   Menu(self.toolbar, tearoff=0),\
                                'edit'      :   Menu(self.toolbar, tearoff=0),
                                'session'   :   Menu(self.toolbar, tearoff=0),
                                'tools'     :   Menu(self.toolbar, tearoff=0)
        }

        # For each dropdown, add commands
        self.tool_dropdown['session'].add_command(label='new',command = lambda x : x)
        self.tool_dropdown['session'].add_command(label='save',command = lambda x : x)
        self.tool_dropdown['session'].add_command(label='load',command = lambda x : x)

        self.tool_dropdown['file'].add_command(label='import',command = lambda x : x)
        self.tool_dropdown['file'].add_command(label='export',command = lambda x : x)
        self.tool_dropdown['file'].add_command(label='replace',command = lambda x : x)


        self.tool_dropdown['edit'].add_command(label='import',command = lambda x : x)
        self.tool_dropdown['edit'].add_command(label='view1',command = lambda  : self.__init__(APP_REF,1))
        self.tool_dropdown['edit'].add_command(label='view2',command = lambda  : self.__init__(APP_REF,2))

        # TOOLS DROPDOWN CREATIONS
        self.tool_dropdown['tools'].add_command(label='word count',command = lambda x : x)
        self.tool_dropdown['tools'].add_command(label='search',command = lambda x : x)
        self.tool_dropdown['tools'].add_command(label='cluster',command = lambda x : x)
        self.tool_dropdown['tools'].add_command(label='topic model',command = lambda x : x)
        self.tool_dropdown['tools'].add_command(label='compare',command = lambda x : x)
        self.tool_dropdown['tools'].add_command(label='compare',command = lambda x : x)

        # add the dropdown to the toolbar
        self.toolbar.add_cascade(label='Session',menu=self.tool_dropdown['session'])
        self.toolbar.add_cascade(label='File',menu=self.tool_dropdown['file'])
        self.toolbar.add_cascade(label='Edit',menu=self.tool_dropdown['edit'])
        self.toolbar.add_cascade(label='Tools',menu=self.tool_dropdown['tools'])

        # add the bar to the window
        APP_REF.window.config(menu=self.toolbar)

    def load_frames(self,APP_REF):
        APP_REF.frames = {
                                    'toolbar'                   :   Frame(APP_REF.window,bg='red'),
                                    'workspace'                 :   Frame(APP_REF.window,bg='blue')\
        }

        APP_REF.viewports['BLOCK1'] = Frame(APP_REF.window)
        if self.n_frames == 2:
            APP_REF.viewports['BLOCK2'] = Frame(APP_REF.window)

        APP_REF.viewports["window1"] = ScrolledText(APP_REF.viewports['BLOCK1'])
        if self.n_frames == 2:
            APP_REF.viewports["window2"] = ScrolledText(APP_REF.viewports['BLOCK2'])

        APP_REF.viewports['topBar'] = Frame(APP_REF.window)
        APP_REF.viewports['topBar'].config(bg='#222222')
        APP_REF.viewports['topBar'].grid(row=0,column=0,padx=0,pady=0,sticky='nsew',columnspan=3)


        APP_REF.viewports['int1'] =  Frame(APP_REF.window)
        APP_REF.viewports['int1'].config(bg='#191d21')

        APP_REF.viewports["int1Button"]   = Button(APP_REF.viewports['int1'], text="import interview", command = lambda : Utilities.import_file(APP_REF,scrolled_text=APP_REF.viewports["window1"]))
        APP_REF.viewports['int1Button'].grid(sticky='nsew',row=0,column=0,padx=0,pady=0)

        APP_REF.viewports['int1'].grid(row=1,column=1,padx=0,pady=0,sticky='nsew')


        APP_REF.viewports['window1'].config(bg='#ebe1be')
        APP_REF.viewports['BLOCK1'].columnconfigure(0,weight=1)
        APP_REF.viewports['BLOCK1'].rowconfigure(0,weight=1)

        APP_REF.viewports['window1'].grid(sticky='nsew',row=0,column=0,padx=4,pady=4)
        APP_REF.viewports["BLOCK1"].grid(row=2,column=1,sticky='nsew', padx=10,pady=5)



        if self.n_frames == 2:
            APP_REF.viewports['int2'] =  Frame(APP_REF.window)
            APP_REF.viewports['int2'].config(bg='#191d21')

            APP_REF.viewports["int2Button"]   = Button(APP_REF.viewports['int2'], text="import interview", command = lambda : Utilities.import_file(APP_REF,scrolled_text=APP_REF.viewports["window2"]))
            APP_REF.viewports['int2Button'].grid(sticky='nsew',row=0,column=0,padx=0,pady=0)


            APP_REF.viewports['int2'].grid(row=1,column=2,padx=0,pady=0,sticky='nsew')



            APP_REF.viewports['window2'].config(bg='#ebe1be')
            APP_REF.viewports['BLOCK2'].columnconfigure(0,weight=1)
            APP_REF.viewports['BLOCK2'].rowconfigure(0,weight=1)

            APP_REF.viewports['window2'].grid(sticky='nsew',row=0,column=0,padx=4,pady=4)
            APP_REF.viewports["BLOCK2"].grid(row=2,column=2,sticky='nsew', padx=10,pady=5)




        #APP_REF.frames['toolbar'].grid(row=0,sticky='ew')
        #APP_REF.frames['workspace'].grid(row=1,sticky='nsew')





        #APP_REF.pdf = tkPDFViewer.ShowPdf()
        #APP_REF.pdf2 = APP_REF.pdf.pdf_view(APP_REF.window, pdf_location = r'D:\Website\reciept.pdf')
        #APP_REF.pdf2.grid(column=1,row=0)
