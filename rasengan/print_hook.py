'''
| Filename    : print_hook.py
| Description : A hook to override stdin and stdout.
| Author      : Pushpendre Rastogi
| Created     : Wed Nov 18 16:19:20 2015 (-0500)
| Last-Updated: Wed Nov 18 16:51:14 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 6
'''
import sys

class PrintHook(object):
    ''' this class gets all output directed to stdout(e.g by
    print statements) and stderr and redirects it to a user defined function
    '''

    def __init__(self):
        ''' out, func and origOut would be delay initialized on Start.
        '''
        self.override = None
        # func would be delayed initialized at start to a function
        # that is called on the text to be printed.
        self.func = None
        self.origOut = None

    @staticmethod
    def default(text):
        '''
        Params
        ------
        text :
        Returns
        -------
        three variables proceed, lineNoMode, newText
        '''
        return text

    def start(self, func=None, override='stdout'):
        if override == 'stdout':
            sys.stdout.flush()
            self.origOut = sys.stdout
            sys.stdout = self
        else:
            sys.stderr.flush()
            self.origOut = sys.stderr
            sys.stderr= self
        self.override = override
        self.func = (func
                     if func
                     else self.TestHook)
        return self

    def stop(self):
        if self.override == 'stdout':
            sys.stdout = self.origOut
        else:
            sys.stderr = self.origOut
        self.func = None


    def write(self, text):
        processed_text = self.func(text)
        self.origOut.write(processed_text)
        return

    def __getattr__(self, name):
        ''' delegate all other methods to origOut so that we
        don't have to override them'''
        return self.origOut.__getattr__(name)
