import os                                           # allows access to filepath
from tkinter import Menu, RAISED, Frame, Label, Button, Entry, Tk, N, S, E, W, Y, X
from tkinter.scrolledtext import *
from Utilities import Utilities
from Utilities import Algorithms
import tkinter 
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

    class work_block:
        def __init__(self,n,components):
            self.frame = n
            self.interview_container = None 
            self.components = components

    def __init__(self, APP_REF,n_frames=2):
        self.n_frames = n_frames
        

        # STYLE SETTINGS 
        self.style = {
                        'block_bg'      : "#a3a7b2",
                        'optionbox_bg'  : "#191d21",
                        'interview_bg'  : '#FFF8ED'
        }


        self.work_block = {}

        #for i in range(n_frames):
        #    self.work_block[i] = work_block(i)

        # Ensure that we start with a clean slate 
        for w in APP_REF.window.winfo_children():
            w.destroy()


        # Create the grid layout of the main frame
        APP_REF.window.columnconfigure(0,weight=1)
        APP_REF.window.columnconfigure(1,weight=1)
        APP_REF.window.columnconfigure(2,weight=1)

        APP_REF.window.rowconfigure(0,weight=1)
        APP_REF.window.rowconfigure(1,weight=1)
        APP_REF.window.rowconfigure(2,weight=10)

        # Configure the background color
        APP_REF.window.config(bg=APP_REF.settings['main_theme_bg'])

        self.load_menu(APP_REF)
        self.load_frames(APP_REF)

    def load_menu(self,APP_REF):
        self.toolbar = Menu(APP_REF.root,relief=RAISED)
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
        #self.tool_dropdown['session'].add_command(label='new',command = lambda x : x)
        self.tool_dropdown['session'].add_command(label='save',command = lambda : Utilities.save_session(APP_REF))
        self.tool_dropdown['session'].add_command(label='load',command = lambda : Utilities.load_session(APP_REF))

        self.tool_dropdown['file'].add_command(label='import',command = lambda : Utilities.import_files(APP_REF))
        #self.tool_dropdown['file'].add_command(label='export',command = lambda x : x)
        #self.tool_dropdown['file'].add_command(label='replace',command = lambda x : x)


        self.tool_dropdown['edit'].add_command(label='import',command = lambda : Utilities.import_files(APP_REF))
        self.tool_dropdown['edit'].add_command(label='view1',command = lambda  : self.__init__(APP_REF,n_frames=1))
        self.tool_dropdown['edit'].add_command(label='view2',command = lambda  : self.__init__(APP_REF,n_frames=2))

        # TOOLS DROPDOWN CREATIONS
        #self.tool_dropdown['tools'].add_command(label='word count',command = lambda x : x)
        self.tool_dropdown['tools'].add_command(label='classify',command = lambda : APP_REF.data['models']['classify'].run(APP_REF))
        self.tool_dropdown['tools'].add_command(label='cluster docs',command = lambda : APP_REF.data['models']['DocClusterer'].run(APP_REF))
        self.tool_dropdown['tools'].add_command(label='topic model',command = lambda : APP_REF.data['models']['topicMultiModel'].run(APP_REF))
        #self.tool_dropdown['tools'].add_command(label='compare',command = lambda x : x)
        #self.tool_dropdown['tools'].add_command(label='compare',command = lambda x : x)

        # add the dropdown to the toolbar
        self.toolbar.add_cascade(label='Session',menu=self.tool_dropdown['session'])
        self.toolbar.add_cascade(label='File',menu=self.tool_dropdown['file'])
        self.toolbar.add_cascade(label='Edit',menu=self.tool_dropdown['edit'])
        self.toolbar.add_cascade(label='Tools',menu=self.tool_dropdown['tools'])

        # add the bar to the window
        APP_REF.root.config(menu=self.toolbar)

    def load_frames(self,APP_REF):
        APP_REF.frames = {
                                    'toolbar'                   :   Frame(APP_REF.window,bg='red'),
                                    'workspace'                 :   Frame(APP_REF.window,bg='blue'),

        }

        for i in range(self.n_frames):
            self.work_block[i] = self.build_block(APP_REF,i,WorkSession.work_block(i,{}))

    def build_block(self, APP_REF, block_num, this_block):

        items = {}

        # Create and configure the base frame 
        block = Frame(APP_REF.window)
        block.config(bg=self.style['block_bg']) 
        block.columnconfigure(0,weight=1)
        block.rowconfigure(0,weight=1)
        block.rowconfigure(1,weight=2)

        # Create the scrolled text portion
        interview_container = ScrolledText(block, font=(APP_REF.settings['font'],APP_REF.settings['text_size']))
        interview_container.config(bg=self.style['interview_bg'])
        
        # Create the options bar above the interview 
        options_bar = Frame(block)         
        options_bar.config(bg=self.style['optionbox_bg'])

        # Buttons
        import_button = Button(options_bar, text="import interview", command = lambda : Utilities.import_file(APP_REF,work_block=this_block))
        import_button.grid(sticky='nsew',row=0,column=0,padx=0,pady=0)

        cluster_button = Button(options_bar, text="cluster interview", command = lambda : APP_REF.data['models']['classify'].run(APP_REF))
        cluster_button.grid(sticky='nsew',row=0,column=1,padx=0,pady=0)

        topic_button = Button(options_bar, text="topic multi interview", command = lambda : APP_REF.data['models']['topicMultiModel'].run(APP_REF))
        topic_button.grid(sticky='nsew',row=0,column=2,padx=0,pady=0)

        topic_button2 = Button(options_bar, text="topic sparse model", command = lambda : APP_REF.data['models']['topicSparseModel'].run(APP_REF))
        topic_button2.grid(sticky='nsew',row=0,column=5,padx=0,pady=0)


        clusterD_button = Button(options_bar, text="cluster all interviews", command = lambda : Algorithms.run_alg_in_window(APP_REF,"Doc Cluster"))
        clusterD_button.grid(sticky='nsew',row=0,column=3,padx=0,pady=0)

        gpt_button = Button(options_bar, text="run GPT", command = lambda : APP_REF.data['models']['gpt'].run(APP_REF))
        gpt_button.grid(sticky='nsew',row=0,column=4,padx=0,pady=0)

        options_bar.grid(row=0,column=0,padx=10,pady=0,sticky='nsew')
        interview_container.grid(sticky='nsew',row=1,column=0,padx=10,pady=4,ipadx=10)
        block.grid(row=1,column=block_num,sticky='nsew', padx=10,pady=5, rowspan=2)


        APP_REF.viewports[f"BLOCK{block_num}"] = block 
        items[f"BLOCK{block_num}"] = block

        APP_REF.viewports[f"OPTIONSBAR{block_num}"] = options_bar
        items[f"OPTIONSBAR{block_num}"] = options_bar

        APP_REF.viewports[f"INTERVIEW{block_num}"] = interview_container
        items[f"INTERVIEW{block_num}"] = interview_container
  
        APP_REF.viewports[f"IMBUTTON{block_num}"] = import_button
        items[f"IMBUTTON{block_num}"] = import_button

        APP_REF.viewports[f"CLBUTTON{block_num}"] = cluster_button
        items[f"CLBUTTON{block_num}"] = cluster_button

        APP_REF.viewports[f"TPBUTTON{block_num}"] = topic_button
        items[f"TPBUTTON{block_num}"] = topic_button

        APP_REF.viewports[f"GPBUTTON{block_num}"] = gpt_button
        items[f"GPBUTTON{block_num}"] = gpt_button

        APP_REF.viewports[f"TP2BUTTON{block_num}"] = topic_button2
        items[f"TP2BUTTON{block_num}"] = topic_button2
        

        APP_REF.viewports[f"CLDBUTTON{block_num}"] = clusterD_button
        items[f"CLDBUTTON{block_num}"] = clusterD_button
        
        this_block.components = items
        this_block.interview_container = interview_container
        return this_block



