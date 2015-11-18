'''
| Filename    : print_hook.py
| Description : A hook to override stdin and stdout.
| Author      : Pushpendre Rastogi
| Created     : Wed Nov 18 16:19:20 2015 (-0500)
| Last-Updated: Wed Nov 18 17:20:45 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 14
'''
import sys

class PrintHook(object):
    ''' this class gets all output directed to stdout(e.g by
    print statements) and stderr and redirects it to a user defined function
    out, func and origOut would be delay initialized on Start.
    '''
    override = None
    func = None
    origOut = None
    already_started = 0

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
        if self.already_started:
            # If we have already overridden an output, then don't
            # overide something else till restore the original back.
            return self
        else:
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
                         else PrintHook.default)
        return self

    def stop(self):
        if self.override == 'stdout':
            sys.stdout = self.origOut
        else:
            sys.stderr = self.origOut
        self.func = None
        self.already_started = 0

    def write(self, text):
        processed_text = self.func(text)
        self.origOut.write(processed_text)
        return

    def __getattr__(self, name):
        ''' delegate all other methods to origOut so that we
        don't have to override them'''

        return getattr(self.origOut, name)
