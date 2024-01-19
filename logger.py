import logging,os,time

class Logging():

    def make_log_dir(self, dirname='./logs'):
        now_dir = os.path.dirname(__file__)
        father_path = os.path.split(now_dir)[0]
        print(father_path)
        # path = os.path.join(father_path,dirname)
        # path = os.path.normpath(path)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        return dirname

    def get_log_filename(self, log_path): 
        filename = "{}.log".format(time.strftime("%Y-%m-%d",time.localtime()))
        filename = os.path.join(self.make_log_dir(log_path),filename)
        # filename = os.path.normpath(filename)
        print(filename)
        return filename

    def log(self, log_path, level='DEBUG'): 
        logger = logging.getLogger()
        levle = getattr(logging,level)
        logger.setLevel(level)
        if not logger.handlers: 
            sh = logging.StreamHandler()
            fh = logging.FileHandler(filename=self.get_log_filename(log_path),mode='a',encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s-%(levelname)s-%(filename)s-Line:%(lineno)d-Message:%(message)s")
            sh.setFormatter(fmt=fmt)
            fh.setFormatter(fmt=fmt)
            logger.addHandler(sh)
            logger.addHandler(fh)
        return logger 

