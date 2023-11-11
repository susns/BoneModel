import glob
import os
from getinfo import loadMeshes, getVerticesAndFaces
import numpy as np

# 返回文件路径名和前缀名
def get_all(dirpath, suffix='.ply'):
    files = glob.glob(os.path.join(dirpath, "*" + suffix))
    names = [os.path.basename(f) for f in files]
    return files, names

# 将目录下的obj文件转换成ply
def transformTxt(dirpath, surfix, targetDir=None):
    files, names = get_all(dirpath, surfix)
    meshes = loadMeshes(files)
    vertices = []
    for i, mesh in enumerate(meshes):
        v, _ = getVerticesAndFaces(mesh)
        vertices.append(v)

    actualDir = targetDir
    if targetDir is None:
        if os.path.exists(os.path.join(dirpath, "plys")) == False:
            os.mkdir(os.path.join(dirpath, "plys"))
        actualDir = os.path.join(dirpath, "plys")

    for i in range(len(vertices)):
        filename = os.path.join(actualDir, os.path.basename(names[i]).split(".")[0] + ".txt")
        np.savetxt(filename, vertices[i], delimiter=" ")


if __name__ == "__main__":
    root_dir = "/Volumes/mnt/Documents/Data/fixed_data/forHust/"
    transformTxt(os.path.join(root_dir, "sample"), '.ply')
