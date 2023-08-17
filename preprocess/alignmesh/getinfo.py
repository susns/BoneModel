import vtk
import numpy as np
import tqdm
from vtk.util.numpy_support import vtk_to_numpy
import glob


def createICP(reference, source, iterations):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(reference)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(iterations)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    return icp


def applyTransform(source, icp):
    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(source)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()
    return icpTransformFilter.GetOutput()


# 计算mesh1 和 mesh2 之间的距离
# method： 'PointToCell' 或 'PointToPoint'
def getDistance(mesh1, mesh2, method):
    distance = vtk.vtkHausdorffDistancePointSetFilter()

    if method == 'PointToCell':
        celllocator1 = vtk.vtkCellLocator()
        celllocator1.SetDataSet(mesh1)
        celllocator1.BuildLocator()
        mesh1.SetCellLocator(celllocator1)

        celllocator2 = vtk.vtkCellLocator()
        celllocator2.SetDataSet(mesh2)
        celllocator2.BuildLocator()
        mesh2.SetCellLocator(celllocator2)

        distance.SetInputData(0, mesh1)
        distance.SetInputData(1, mesh2)
        distance.Update()
    elif method == 'PointToPoint':
        distance.SetTargetDistanceMethod(1)

        kdlocator1 = vtk.vtkKdTreePointLocator()
        kdlocator1.SetDataSet(mesh1)
        kdlocator1.BuildLocator()
        mesh1.SetPointLocator(kdlocator1)

        kdlocator2 = vtk.vtkKdTreePointLocator()
        kdlocator2.SetDataSet(mesh2)
        kdlocator2.BuildLocator()
        mesh2.SetPointLocator(kdlocator2)

        distance.SetInputData(0, mesh1)
        distance.SetInputData(1, mesh2)
        distance.Update()
    return distance.GetHausdorffDistance()


def findReferenceMesh(meshes):
    pairs = []
    for i in range(len(meshes)):
        j = i + 1
        while j < len(meshes):
            pairs.append([i, j])
            j = j + 1

    means = np.zeros(len(meshes))
    for i in tqdm.trange(len(pairs)):
        pair = pairs[i]
        mesh1, mesh2 = vtk.vtkPolyData(), vtk.vtkPolyData()
        mesh1.DeepCopy(meshes[pair[0]])
        mesh2.DeepCopy(meshes[pair[1]])
        icp = createICP(mesh1, mesh2, 10)
        transformd = applyTransform(mesh2, icp)
        result = getDistance(mesh1, transformd, 'PointToCell')
        means[pair[0]] += result / len(meshes)
        means[pair[1]] += result / len(meshes)
    return np.argmin(means)


def loadMeshes(mesh_files):
    suffix = mesh_files[0].split('.')[1]

    def getReader():
        if suffix == 'vtk':
            return vtk.vtkPolyDataReader()
        elif suffix == 'obj':
            return vtk.vtkOBJReader()
        elif suffix == 'ply':
            return vtk.vtkPLYReader()

    meshes = []
    for file in mesh_files:
        reader = getReader()
        reader.SetFileName(file)
        reader.Update()
        meshes.append(reader.GetOutput())
    return meshes


def getVerticesAndFaces(mesh):
    points = mesh.GetPoints().GetData()
    vertices = vtk_to_numpy(points)

    ids = mesh.GetPolys()
    faces = []
    ids.InitTraversal()
    while True:
        pointsIds = vtk.vtkIdList()
        if ids.GetNextCell(pointsIds):
            face = np.zeros(pointsIds.GetNumberOfIds())
            for i in range(pointsIds.GetNumberOfIds()):
                face[i] = pointsIds.GetId(i)
            faces.append(face)
        else:
            break
    return vertices, np.array(faces, dtype=int)