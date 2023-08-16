import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import KDTree



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


def ICPmanu_allign2(target, source):
    kdtree1, kdtree2 = KDTree(target), KDTree(source)
    dis, ids = kdtree1.query(source)
    IDX1 = np.concatenate((ids[:, np.newaxis], dis[:, np.newaxis]), axis=1)
    dis, ids = kdtree2.query(target)
    IDX2 = np.concatenate((ids[:, np.newaxis], dis[:, np.newaxis]), axis=1)
    IDX1 = np.concatenate((IDX1, np.arange(len(source))[:, np.newaxis]), axis=1)
    IDX2 = np.concatenate((IDX2, np.arange(len(target))[:, np.newaxis]), axis=1)

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


def rigidICP(target, source):
    Prealligned_source, Prealligned_target, transformtarget = Preall(target, source)
    error = 0
    new_error, Reallignedsourcetemp = ICPmanu_allign2(Prealligned_target, Prealligned_source)

    while np.abs(new_error - error) > 0.000001:
        error = new_error
        new_error, Reallignedsourcetemp = ICPmanu_allign2(Prealligned_target, Reallignedsourcetemp)
    error = new_error
    Reallignedsource = Reallignedsourcetemp @ transformtarget['rotation'] + np.tile(transformtarget['translation'][:3],
                                                                                    (len(Reallignedsourcetemp), 1))

    # 先用icp将目标形状计算出来，然后用procrustes计算变换
    _, Reallignedsource, transform = procrustes(Reallignedsource, source)
    return error, Reallignedsource, transform


# 使用icp算法对齐形状
def ICPmanu_allignSSM(vnew, MEAN3d, estimate, BTXX, BTXY, BTXZ, nmodes):
    # estimate 和 vnew 都是(M, 3)
    # 找到vnew中 离estimate 最近的点
    # IDX 第一列表示X的索引，第二列表示距离
    kdtree1, kdtree2 = KDTree(vnew), KDTree(estimate)
    dis, ids = kdtree1.query(estimate)
    IDX1 = np.concatenate((ids[:, np.newaxis], dis[:, np.newaxis]), axis=1)
    dis, ids = kdtree2.query(vnew)
    IDX2 = np.concatenate((ids[:, np.newaxis], dis[:, np.newaxis]), axis=1)

    # IDX 第三列表示Y的索引 (M, 3)
    IDX1 = np.concatenate((IDX1, np.arange(len(estimate))[:, np.newaxis]), axis=1)
    IDX2 = np.concatenate((IDX2, np.arange(len(vnew))[:, np.newaxis]), axis=1)

    # Datasetsource 和 Datasettarget 都是(2M, 3)
    # Source中的点来自estimate， target中的点来自vnew
    Datasetsource = np.concatenate((MEAN3d[IDX1[:, 2].astype(int)], MEAN3d[IDX2[:, 0].astype(int)]))
    Datasettarget = np.concatenate((vnew[IDX1[:, 0].astype(int)], vnew))

    # 将Datasetsource转化为列向量
    MEANaugmented = Datasetsource.reshape(3 * len(Datasetsource), 1)

    # 垂直拼接特征向量（提取对应于Datasource中的特征向量），(6M, n)
    BTXaugmented = np.concatenate((BTXX[IDX1[:, 2].astype(int)], BTXX[IDX2[:, 0].astype(int)],
                                   BTXY[IDX1[:, 2].astype(int)], BTXY[IDX2[:, 0].astype(int)],
                                   BTXZ[IDX1[:, 2].astype(int)], BTXZ[IDX2[:, 0].astype(int)]))

    # (6M, 1)
    Zssm = (Datasettarget - Datasetsource).reshape(len(BTXaugmented), 1)
    # pinv 求伪逆 A .B .A = A (A不必为方阵或者满秩)
    # 假设nmodes = x, 则PI为(x, 6M)
    PI = np.linalg.pinv(BTXaugmented[:, :nmodes])
    # b = pi * (X - S')
    EstimatedModes = PI @ Zssm
    # 拼接特征矩阵（只包含目标modes）
    BTX = np.concatenate(([BTXX[:, :nmodes], BTXY[:, :nmodes], BTXZ[:, :nmodes]]))
    tempestimate = MEAN3d.reshape(len(MEAN3d) * 3, 1) + BTX @ EstimatedModes

    # 更新estimate
    # (M,3)
    estimate = tempestimate.reshape(int(len(tempestimate) / 3), 3)

    kdtree3 = KDTree(estimate)
    dis, ids = kdtree1.query(estimate)
    IDX1 = np.concatenate((ids[:, np.newaxis], dis[:, np.newaxis]), axis=1)
    dis, ids = kdtree3.query(vnew)
    IDX2 = np.concatenate((ids[:, np.newaxis], dis[:, np.newaxis]), axis=1)
    IDX1 = np.concatenate((IDX1, np.arange(len(estimate))[:, np.newaxis]), axis=1)
    IDX2 = np.concatenate((IDX2, np.arange(len(vnew))[:, np.newaxis]), axis=1)

    # 度量新的estimate 与 vnew 的差异
    m1 = IDX2.mean(axis=0)[1]
    s1 = IDX2.std(axis=0)[1]
    IDX1 = IDX1[IDX1[:, 1] < m1 + 1.96 * s1]

    Datasetsource = np.concatenate((estimate[IDX1[:, 2].astype(int)], estimate[IDX2[:, 0].astype(int)]))
    Datasettarget = np.concatenate((vnew[IDX1[:, 0].astype(int)], vnew))

    # 对Datasettarget 应用procrustes 分析进行变换
    error, Reallignedtarget1, _ = procrustes(Datasetsource, Datasettarget)
    Reallignedtarget = Reallignedtarget1[len(IDX1):]
    return error, Reallignedtarget, estimate, EstimatedModes


def SSMfitter(MEAN, ssmV, V, nmodes):
    '''
    input:
    MEAN: (M, 3)
    ssmV: (3M, n)
    V:    (M,  3)
    nmodes: scaler
    output:
    EstimatedModes: b
    '''
    # p, q = MEAN.shape
    # s = int(p / 3)
    # MEANX = MEAN[:s]
    # MEANY = MEAN[s:2 * s]
    # MEANZ = MEAN[2 * s:]

    # 拼接mean 为(M, 3)
    # MeanParentbone = np.concatenate((MEANX, MEANY, MEANZ), axis=1)
    MeanParentbone = MEAN
    # 应用ICP刚性对齐 对准mean 和 V
    error, ReallignedVtarget, transform = rigidICP(MeanParentbone, V)

    # 获取X, Y, Z 三个坐标轴的特征向量
    s = MEAN.shape[0]
    BTXX = ssmV[:s]
    BTXY = ssmV[s:2 * s]
    BTXZ = ssmV[2 * s:]

    estimate = MeanParentbone
    error, Reallignedtargettemp, estimate, EstimatedModes = ICPmanu_allignSSM(ReallignedVtarget, MeanParentbone,
                                                                              estimate, BTXX, BTXY, BTXZ, nmodes)

    while True:
        new_error, Reallignedtargettemp, estimate, EstimatedModes = ICPmanu_allignSSM(Reallignedtargettemp,
                                                                                      MeanParentbone, estimate, BTXX,
                                                                                      BTXY, BTXZ, nmodes)
        if np.abs(new_error - error) <= 0.00001:
            break
        error = new_error
    error = new_error

    SSMfit = estimate
    _, ReallignedV, transform = procrustes(Reallignedtargettemp, V)

    kdtree = KDTree(ReallignedV)
    dis, ids = kdtree.query(SSMfit)
    landmarks = ReallignedV[ids]

    position_error = np.sqrt(np.sum((landmarks - SSMfit) ** 2, axis=1))
    RMSerror = np.sqrt(np.mean(position_error))
    return RMSerror, ReallignedV, transform, SSMfit, EstimatedModes, position_error


def change_ssmV(origin):
    '''
    Args:
        origin: from OriginPCA::self.p, (n, 3M)

    Returns: used for SSMfitter::ssmV, (3M, n), columns group by x, y, z

    '''
    n, m = origin.shape
    i = 0
    s = int(m/3)
    BTX = []
    BTY = []
    BTZ = []
    while i < m:
        BTX.append(origin[:, i])
        BTY.append(origin[:, i+1])
        BTZ.append(origin[:, i+2])
        i += 3
    BTX, BTY, BTZ = np.array(BTX), np.array(BTY), np.array(BTZ)
    BTX = np.concatenate((BTX, BTY, BTZ))
    return BTX