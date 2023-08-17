import glob
import os


# 返回文件路径名和前缀名
def get_all(dirpath, suffix='.ply'):
    files = glob.glob(os.path.join(dirpath, "*" + suffix))
    names = [os.path.basename(f) for f in files]
    return files, names