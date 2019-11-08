# uncompyle6 version 3.5.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\utils.py
# Compiled at: 2018-06-19 19:17:07
# Size of source mod 2**32: 719 bytes
import time

class Timer:

    def __init__(self):
        self.start_time = time.time()
        self.job = None

    def start(self, job, verbal=False):
        self.job = job
        self.start_time = time.time()
        if verbal:
            print('[I] {job} started.'.format(job=(self.job)))

    def stop(self):
        if self.job is None:
            return
        elapsed_time = time.time() - self.start_time
        print('[I] {job} finished in {elapsed_time:0.3f} s.'.format(job=(self.job),
          elapsed_time=elapsed_time))
        self.job = None


class Log:
    verbose = True

    @staticmethod
    def log(text):
        if Log.verbose:
            print(text)