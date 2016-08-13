#!/usr/bin/env python
'''
| Filename    : function_words.py
| Description : List of function words.
| Author      : Pushpendre Rastogi
| Created     : Sat Aug  6 16:28:14 2016 (-0400)
| Last-Updated: Fri Aug 12 18:05:01 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 4
'''
import os


def read_commented_list_file(fn, comment='//'):
    with open(fn) as f:
        for row in f:
            if not row.startswith(comment):
                yield row


def get_function_words():
    l = []
    _dir = 'English_Function_Words_Set'
    for fn in ('EnglishAuxiliaryVerbs.txt EnglishConjunctions.txt '
               'EnglishDeterminers.txt EnglishPrepositions.txt '
               'EnglishPronouns.txt EnglishQuantifiers.txt').split():
        for row in read_commented_list_file(
                os.path.join(os.path.dirname(__file__), _dir, fn)):
            l.append(row.strip())
    return l


def get_auxiliary_verbs():
    return ('be am are is was were being been can could dare do does did have '
            'has had having may might must need ought shall should will would').split()
