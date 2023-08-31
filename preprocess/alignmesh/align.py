import pathlib

import numpy as np
from scipy.spatial import KDTree
import trimesh
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import tqdm
import os
import preprocess.alignmesh.getinfo as getinfo
import preprocess.alignmesh.tools as tools


# 对齐mesh —— nonrigidICPV2
def procrustes(X, Y, scaling=True, reflection='best'):
    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


def exceptIndice(IDX, indices):
    temp = []
    for line in IDX:
        if line[0] not in indices:
            temp.append(line)
    return np.array(temp)


def ICPmanu_allign2(target, source, Indices_edgesS, Indices_edgesT):
    kdtree1, kdtree2 = KDTree(target), KDTree(source)
    dis, ids = kdtree1.query(source)
    IDX1 = np.concatenate((ids[:, np.newaxis], dis[:, np.newaxis]), axis=1)
    dis, ids = kdtree2.query(target)
    IDX2 = np.concatenate((ids[:, np.newaxis], dis[:, np.newaxis]), axis=1)
    IDX1 = np.concatenate((IDX1, np.arange(len(source))[:, np.newaxis]), axis=1)
    IDX2 = np.concatenate((IDX2, np.arange(len(target))[:, np.newaxis]), axis=1)

    IDX2 = exceptIndice(IDX2, Indices_edgesS)
    IDX1 = exceptIndice(IDX1, Indices_edgesT)

    m1 = IDX1.mean(axis=0)[1]
    s1 = IDX1.std(axis=0)[1]
    IDX2 = IDX2[IDX2[:, 1] < (m1 + 1.96 * s1)]

    Datasetsource = np.concatenate((source[IDX1[:, 2].astype(int)], source[IDX2[:, 0].astype(int)]))
    Datasettarget = np.concatenate((target[IDX1[:, 0].astype(int)], target[IDX2[:, 2].astype(int)]))
    error, Realignedsource, transform = procrustes(Datasettarget, Datasetsource)
    Reallignedsource = transform['scale'] * source @ transform['rotation'] + np.tile(transform['translation'][:3],
                                                                                     (len(source), 1))
    return error, Reallignedsource


def Preall(target, source):
    '''
    执行预先对齐
    [coeff, score, latent] = pca(data);
    ==
    pca = PCA()
    coeff = pca.fit_transform(data)
    score = pca.components_
    latent = pca.explained_variance_
    '''
    R = np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0, ], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    ])

    # pca 和 matlab中不一样
    pca = PCA()
    Prealligned_source = pca.fit_transform(source)
    Prealligned_target = pca.fit_transform(target)

    Maxtarget = Prealligned_source.max(axis=0) - Prealligned_source.min(axis=0)
    Maxsource = Prealligned_target.max(axis=0) - Prealligned_target.min(axis=0)
    D = Maxtarget / Maxsource
    D = np.array([[D[0], 0, 0], [0, D[1], 0], [0, 0, D[2]]])
    RTY = Prealligned_source @ D

    MM = []
    for T in R:
        T = RTY @ T
        kdtree = KDTree(T)
        DD, bb = kdtree.query(Prealligned_target)
        MM.append(DD.sum())
    MM = np.array(MM)
    M, I = MM.min(), MM.argmin()
    T = R[I]
    Prealligned_source = Prealligned_source @ T
    _, _, transformtarget = procrustes(target, Prealligned_target)
    return Prealligned_source, Prealligned_target, transformtarget


def rigidICP(target, source, Indices_edgesS, Indices_edgesT):
    # 默认执行预对齐
    Prealligned_source, Prealligned_target, transformtarget = Preall(target, source)
    error = 0
    new_error, Reallignedsourcetemp = ICPmanu_allign2(Prealligned_target, Prealligned_source, Indices_edgesS,
                                                      Indices_edgesT)

    while np.abs(new_error - error) > 0.000001:
        error = new_error
        new_error, Reallignedsourcetemp = ICPmanu_allign2(Prealligned_target, Reallignedsourcetemp, Indices_edgesS,
                                                          Indices_edgesT)
    error = new_error
    Reallignedsource = Reallignedsourcetemp @ transformtarget['rotation'] + np.tile(transformtarget['translation'][:3],
                                                                                    (len(Reallignedsourcetemp), 1))

    # 先用icp将目标形状计算出来，然后用procrustes计算变换
    _, Reallignedsource, transform = procrustes(Reallignedsource, source)
    return error, Reallignedsource, transform


def definecutoff(vold, fold):
    fk1 = fold[:, 0].astype(int)
    fk2 = fold[:, 1].astype(int)
    fk3 = fold[:, 2].astype(int)
    numverts = vold.shape[0]
    numfaces = fold.shape[0]
    D1 = np.sqrt(np.sum((vold[fk1] - vold[fk2]) ** 2, axis=1))
    D2 = np.sqrt(np.sum((vold[fk1] - vold[fk3]) ** 2, axis=1))
    D3 = np.sqrt(np.sum((vold[fk2] - vold[fk3]) ** 2, axis=1))

    aver = np.mean(np.array([D1, D2, D3]))
    stdevui = np.std(np.array([D1, D2]))
    return aver, stdevui


def detectedges(V, F):
    fk1 = F[:, 0]
    fk2 = F[:, 1]
    fk3 = F[:, 2]

    ed1 = np.sort(np.concatenate((fk1[:, np.newaxis], fk2[:, np.newaxis]), axis=1), axis=1)
    ed2 = np.sort(np.concatenate((fk1[:, np.newaxis], fk3[:, np.newaxis]), axis=1), axis=1)
    ed3 = np.sort(np.concatenate((fk2[:, np.newaxis], fk3[:, np.newaxis]), axis=1), axis=1)

    ed = np.concatenate((ed1, ed2, ed3))
    _, ia = np.unique(ed, axis=0, return_index=True)
    esingle = ed[ia]
    mask = np.ones(len(ed), dtype=bool)
    mask[ia] = False
    edouble = ed[mask]
    C = np.setdiff1d(esingle, edouble)
    return C


def find(IDX, indices, col):
    mask = np.zeros(len(IDX), dtype=bool)
    for i, line in enumerate(IDX):
        if line[col] in indices:
            mask[i] = True
    return mask


def nonrigidICPv2(targetV, sourceV, targetF, sourceF, iterations):
    # 检测重复点
    targetV, indices = np.unique(targetV, return_inverse=True, axis=0)
    targetF = indices[targetF.astype(int)]

    cutoff, _ = definecutoff(sourceV, sourceF)

    # 检测自由边，正常情况下应返回空集
    # mesh中每条边不应该只出现一次
    Indices_edgesS = detectedges(sourceV, sourceF)
    Indices_edgesT = detectedges(targetV, targetF)

    #     print('---Start rigidICP---')
    error, sourceV, transform = rigidICP(targetV, sourceV, Indices_edgesS, Indices_edgesT)

    p = sourceV.shape[0]

    xlimm = np.min(sourceV[:, 0])
    xlimM = np.max(sourceV[:, 0])
    Lx = np.abs(xlimM - xlimm)
    ylimm = np.min(sourceV[:, 1])
    ylimM = np.max(sourceV[:, 1])
    Ly = np.abs(ylimM - ylimm)
    zlimm = np.min(sourceV[:, 2])
    zlimM = np.max(sourceV[:, 2])
    Lz = np.abs(zlimM - zlimm)
    minL = np.min([Lx, Ly, Lz])

    #     print('---General deformation---')
    episilon = 0.00001
    kernel1 = np.arange(start=1.5, stop=1 - episilon, step=-(0.5 / iterations))
    kernel2 = np.arange(start=2.1, stop=2.4 + episilon, step=0.3 / iterations)
    for i in range(iterations):
        # 求seedingmatrix 对应于sourceV中的中心点
        nrseedingpoints = np.round(10 ** kernel2[i])
        lengthseeding = minL / (nrseedingpoints) ** (1 / 3)
        xseeding = np.arange(xlimm, xlimM + episilon, lengthseeding)
        yseeding = np.arange(ylimm, ylimM + episilon, lengthseeding)
        zseeding = np.arange(zlimm, zlimM + episilon, lengthseeding)
        seedingmatrix = np.zeros((len(xseeding) * len(yseeding) * len(zseeding), 3))
        seedingmatrix[:, 2] = np.tile(zseeding, len(xseeding) * len(yseeding))
        tempy = np.tile(yseeding, len(zseeding))
        seedingmatrix[:, 1] = np.tile(tempy, len(xseeding))
        tempx = np.tile(xseeding, len(zseeding) * len(yseeding))
        seedingmatrix[:, 0] = tempx
        kdtree = KDTree(seedingmatrix)
        d, idx = kdtree.query(sourceV)
        tempidx = np.unique(idx)
        tempseed = seedingmatrix[tempidx]
        seedingmatrix = tempseed
        q = len(seedingmatrix)

        kdtree1, kdtree2 = KDTree(targetV), KDTree(sourceV)
        dis, ids = kdtree1.query(sourceV)
        IDX1 = np.concatenate((ids[:, np.newaxis], dis[:, np.newaxis]), axis=1)
        dis, ids = kdtree2.query(targetV)
        IDX2 = np.concatenate((ids[:, np.newaxis], dis[:, np.newaxis]), axis=1)
        IDX1 = np.concatenate((IDX1, np.arange(len(sourceV))[:, np.newaxis]), axis=1)
        IDX2 = np.concatenate((IDX2, np.arange(len(targetV))[:, np.newaxis]), axis=1)
        IDX1 = exceptIndice(IDX1, Indices_edgesT)
        IDX2 = exceptIndice(IDX2, Indices_edgesS)

        sourcepartial = sourceV[IDX1[:, 2].astype(int)]
        targetpartial = targetV[IDX2[:, 2].astype(int)]

        kdtreea, kdtreeb = KDTree(targetpartial), KDTree(sourcepartial)
        ds, IDXS = kdtreea.query(sourcepartial)
        dt, IDXT = kdtreeb.query(targetpartial)

        ppartial = len(sourcepartial)

        D = cdist(sourcepartial, seedingmatrix)
        gamma = 1 / (2 * D.mean()) ** kernel1[i]
        Datasetsource = np.concatenate((sourcepartial, sourcepartial[IDXT]))
        Datasettarget = np.concatenate((targetpartial[IDXS], targetpartial))
        Datasetsource2 = np.concatenate((D, D[IDXT]))
        vectors = Datasettarget - Datasetsource
        r = len(vectors)

        tempy1 = np.exp(-gamma * Datasetsource2 ** 2)
        tempy2 = np.zeros((3 * r, 3 * q))
        tempy2[:r, :q] = tempy1
        tempy2[r:2 * r, q:2 * q] = tempy1
        tempy2[2 * r:, 2 * q:] = tempy1
        lambda_ = 0.001
        ppi = np.linalg.inv(tempy2.T @ tempy2 + lambda_ * np.eye(3 * q)) @ tempy2.T
        modes = ppi @ np.reshape(vectors, (3 * r, 1))

        D2 = D = cdist(sourceV, seedingmatrix)
        gamma2 = 1 / (2 * D.mean()) ** kernel1[i]
        tempyfull1 = np.exp(-gamma2 * D2 ** 2)
        tempyfull2 = np.zeros((3 * p, 3 * q))
        tempyfull2[:p, :q] = tempyfull1
        tempyfull2[p:2 * p, q:2 * q] = tempyfull1
        tempyfull2[2 * p:, 2 * q:] = tempyfull1

        test2 = tempyfull2 @ modes
        test2 = np.reshape(test2, (int(len(test2) / 3), 3))
        sourceV = sourceV + test2
        error, sourceV, transform = rigidICP(targetV, sourceV, Indices_edgesS, Indices_edgesT)

    #     print('---local deformation---')
    kk = 12 + iterations

    # 计算目标表面每个点的法向量
    TR = trimesh.Trimesh(vertices=targetV, faces=targetF, process=False)
    normalsT = TR.vertex_normals * cutoff

    TRS = trimesh.Trimesh(vertices=sourceV, faces=sourceF, process=False)
    normalsS = TRS.vertex_normals * cutoff

    # 根据距离、法线相似性找sourceV上邻近点
    estimateS = np.concatenate((sourceV, normalsS), axis=1)
    kdtree3 = KDTree(estimateS)
    Dsource, IDXsource = kdtree3.query(estimateS, kk)

    # 判断法线方向
    kdtree4 = KDTree(targetV)
    Dcheck, IDXcheck = kdtree4.query(sourceV)
    testpos = np.sum((normalsS - normalsT[IDXcheck]) ** 2)
    testneg = np.sum((normalsS + normalsT[IDXcheck]) ** 2)
    if testneg < testpos:
        normalsT = -normalsT
        targetF[:, [1, 2]] = targetF[:, [2, 1]]

    for ddd in range(1, iterations + 1):
        k = kk - ddd
        TRS = trimesh.Trimesh(vertices=sourceV, faces=sourceF, process=False)
        normalsS = TRS.vertex_normals * cutoff

        sumD = np.sum(Dsource[:, :k], axis=1)[:, np.newaxis]
        sumD2 = np.tile(sumD, (1, k))
        sumD3 = sumD2 - Dsource[:, :k]
        sumD2 = sumD2 * (k - 1)
        weights = sumD3 / sumD2

        estimateT = np.concatenate((targetV, normalsT), axis=1)
        kdtree5 = KDTree(estimateT)
        Dtarget, IDXtarget = kdtree5.query(np.concatenate((sourceV, normalsS), axis=1), 3)
        pp1 = len(targetV)

        # 矫正target中的空洞
        if len(Indices_edgesT) != 0:
            correctionfortargetholes1 = find(IDXtarget, Indices_edgesT, 0)
            targetV = np.concatenate((targetV, sourceV[correctionfortargetholes1]))
            IDXtarget[correctionfortargetholes1, 0] = pp1 + np.arange(np.sum(correctionfortargetholes1))
            Dtarget[correctionfortargetholes1, 0] = 0.00001

            correctionfortargetholes2 = find(IDXtarget, Indices_edgesT, 1)
            pp = len(targetV)
            targetV = np.concatenate((targetV, sourceV[correctionfortargetholes2]))
            IDXtarget[correctionfortargetholes2, 1] = pp + np.arange(np.sum(correctionfortargetholes2))
            Dtarget[correctionfortargetholes2, 1] = 0.00001

            correctionfortargetholes3 = find(IDXtarget, Indices_edgesT, 2)
            pp = len(targetV)
            targetV = np.concatenate((targetV, sourceV[correctionfortargetholes3]))
            IDXtarget[correctionfortargetholes3, 2] = pp + np.arange(np.sum(correctionfortargetholes3))
            Dtarget[correctionfortargetholes3, 2] = 0.00001

        summD = np.sum(Dtarget, axis=1)[:, np.newaxis]
        summD2 = np.tile(summD, (1, 3))
        summD3 = summD2 - Dtarget
        weightsm = summD3 / (2 * summD2)
        Targettempset = np.concatenate(((weightsm[:, 0] * targetV[IDXtarget[:, 0], 0])[:, np.newaxis],
                                        (weightsm[:, 0] * targetV[IDXtarget[:, 0], 1])[:, np.newaxis],
                                        (weightsm[:, 0] * targetV[IDXtarget[:, 0], 2])[:, np.newaxis]),
                                       axis=1) + np.concatenate(((weightsm[:, 1] * targetV[IDXtarget[:, 1], 0])[:,
                                                                 np.newaxis],
                                                                 (weightsm[:, 1] * targetV[IDXtarget[:, 1], 1])[:,
                                                                 np.newaxis],
                                                                 (weightsm[:, 1] * targetV[IDXtarget[:, 1], 2])[:,
                                                                 np.newaxis]), axis=1) + np.concatenate(((weightsm[:,
                                                                                                          2] * targetV[
                                                                                                              IDXtarget[
                                                                                                              :,
                                                                                                              2], 0])[:,
                                                                                                         np.newaxis], (
                                                                                                                                  weightsm[
                                                                                                                                  :,
                                                                                                                                  2] *
                                                                                                                                  targetV[
                                                                                                                                      IDXtarget[
                                                                                                                                      :,
                                                                                                                                      2], 1])[
                                                                                                                      :,
                                                                                                                      np.newaxis],
                                                                                                         (weightsm[:,
                                                                                                          2] * targetV[
                                                                                                              IDXtarget[
                                                                                                              :,
                                                                                                              2], 2])[:,
                                                                                                         np.newaxis]),
                                                                                                        axis=1)

        targetV = targetV[:pp1]
        arraymap = dict()
        for i in range(len(sourceV)):
            sourceset = sourceV[IDXsource[i, :k].T]
            targetset = Targettempset[IDXsource[i, :k].T]
            _, _, arraymap[i] = procrustes(targetset, sourceset, scaling=False)

        sourceVapprox = sourceV
        for i in range(len(sourceV)):
            sourceVtemp = np.zeros(sourceV.shape)
            for ggg in range(k):
                id_ = IDXsource[i, ggg]
                sourceVtemp[ggg] = weights[i, ggg] * (
                            arraymap[id_]['scale'] * sourceV[i] @ arraymap[id_]['rotation'] + arraymap[id_][
                        'translation'])
            sourceV[i] = np.sum(sourceVtemp[:k], axis=0)
        sourceV = sourceVapprox + 0.5 * (sourceV - sourceVapprox)

    registered = sourceV
    return registered, targetV, targetF


# 计算均方根误差
def getRMSE(target, source):
    kdtree = KDTree(target)
    dis, ids = kdtree.query(source)
    landmarks = target[ids]

    position_error = np.sqrt(np.sum((landmarks - source) ** 2, axis=1))
    RMSerror = np.sqrt(np.mean(position_error))
    return RMSerror


# 对齐所有形状，对齐后训练集每个形状为相同的面信息
def alignedAllShape(vertices_list, faces_list, reference_id=0):
    for i in tqdm.trange(len(vertices_list)):
        if i != reference_id:
            sourceV = np.copy(vertices_list[reference_id])
            sourceF = np.copy(faces_list[reference_id])
            targetV = vertices_list[i]
            targetF = faces_list[i]
            registered, _, _ = nonrigidICPv2(targetV, sourceV, targetF, sourceF, 10)
            print(getRMSE(targetV, registered))
            vertices_list[i] = registered
    return vertices_list, faces_list[reference_id]


# names表示mesh的文件名列表
def saveMeshes(vertices_list, faces, names, out_dir):
    suffix = names[0].split('.')[1]
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    for i in tqdm.trange(len(vertices_list)):
        basename = os.path.basename(names[i].replace('.' + suffix, '.obj'))
        mesh = trimesh.Trimesh(vertices_list[i], faces)
        # 平滑
        trimesh.smoothing.filter_laplacian(mesh, iterations=10)
        mesh.export(os.path.join(out_dir, basename))


def align(from_path, save_path):
    files, names = tools.get_all(from_path, ".ply")
    meshes = getinfo.loadMeshes(files)
    print('--- find reference ---')
    if 'example.ply' in names:
        reference_id = names.index('example.ply')
    else:
        reference_id = getinfo.findReferenceMesh(meshes)
    print(f'reference {names[reference_id]}')
    vertices_list = []
    faces_list = []
    print('--- extract vertices and faces ---')
    for i, mesh in enumerate(meshes):
        vertices, faces = getinfo.getVerticesAndFaces(mesh)
        vertices_list.append(vertices)
        faces_list.append(faces)
        print(f'{i + 1}/{len(names)} {names[i]} ')
        print(f'vertices_size: {len(vertices)}')
        print(f'faces_size: {len(faces)}')
        print()
    print('--- align all mesh ---')
    aligned_vertices_list, faces = alignedAllShape(vertices_list, faces_list, reference_id)
    saveMeshes(aligned_vertices_list, faces, names, save_path)


if __name__ == "__main__":
    from_path = "../../data/new/sample"
    save_path = "../../data/new/align"
    align(from_path, save_path)
