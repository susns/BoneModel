import json
import os.path

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial.distance import directed_hausdorff

from tools import get_all
from TraditionalPCA import TraditionalPCA
import numpy as np

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from tools import get_all
from TraditionalPCA import TraditionalPCA
import numpy as np
from SAP.optim_one import main as get_optimize_mesh, get_target
from tools import distance_Chamfer


from SAP.generate_one import toNpz
from SAP.generate_one import main as main_generate


class BoneModelVisApp:
    PointCloud = 'PointCloud'
    Mesh = 'Mesh'

    def __init__(self):
        self.pc = o3d.geometry.PointCloud()
        self.mesh = None
        self.mat = None
        self.pca = None
        self.X = None
        self.path = None

        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window("Visualizer", 1000, 600)
        w = self.window
        em = w.theme.font_size

        # 渲染窗口
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # 右侧面板
        self._pannel = self.make_right(em)

        # 布局回调函数
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._pannel)

    # layout
    def make_right(self, em):
        pannel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.254 * em))

        # Create a dir-chooser widget.
        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label("Data Directory:"))
        self._fileedit = gui.TextEdit()
        self._fileedit.enabled = False
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * em)
        filedlgbutton = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)
        fileedit_layout.add_child(filedlgbutton)
        pannel.add_child(fileedit_layout)
        pannel.add_fixed(0.5 * em)

        # show in pc or mesh
        self.switch_PointCloud_Mesh = gui.ToggleSwitch("PointCloud/Mesh")
        self.switch_PointCloud_Mesh.set_on_clicked(self._on_switch_type)
        pannel.add_child(self.switch_PointCloud_Mesh)
        pannel.add_fixed(0.5 * em)
        self.mesh_type = gui.RadioButton(gui.RadioButton.VERT)
        self.mesh_type.set_items(["Static Topology", "Optimization SAP", "Learning SAP"])
        self.mesh_type.set_on_selection_changed(self._on_mesh_type_change)
        pannel.add_child(self.mesh_type)
        pannel.add_fixed(0.5 * em)
        horizlayout = gui.Horiz()
        horizlayout.add_child(gui.Label("Chamfer"))
        self.Chamfer = gui.TextEdit()
        self.Chamfer.enabled = False
        self.Chamfer.text_value = '0.000000'
        horizlayout.add_child(self.Chamfer)
        horizlayout.add_child(gui.Label("Hausdorff"))
        self.Hausdorff = gui.TextEdit()
        self.Hausdorff.enabled = False
        self.Hausdorff.text_value = '0.000000'
        horizlayout.add_child(self.Hausdorff)
        pannel.add_fixed(0.5 * em)
        pannel.add_child(horizlayout)
        computechamfer = gui.Button('Compute Distance')
        computechamfer.set_on_clicked(self._on_compute_chamfer_button)
        computechamfer.horizontal_padding_em = 0.5
        computechamfer.vertical_padding_em = 0
        pannel.add_child(computechamfer)
        pannel.add_fixed(2 * em)

        self.total_num = gui.NumberEdit(gui.NumberEdit.INT)
        self.total_num.set_limits(0, 100)
        self.total_num.int_value = 0
        self.total_num.set_on_value_changed(self._on_n_change)
        lineone = gui.Horiz()
        lineone.add_child(self.total_num)
        lineone.add_child(gui.Label("components"))
        lineone.add_fixed(em)
        linetwo = gui.Horiz()
        linetwo.add_child(gui.Label('Count'))
        self.contribution = gui.TextEdit()
        self.contribution.enabled = False
        self.contribution.text_value = '0.00%'
        linetwo.add_child(self.contribution)
        linetwo.add_child(gui.Label('RMSE'))
        self.RMSE = gui.TextEdit()
        self.RMSE.enabled = False
        self.RMSE.text_value = '0.000mm'
        linetwo.add_child(self.RMSE)
        linetwo.add_fixed(em)
        pannel.add_child(lineone)
        pannel.add_child(linetwo)
        pannel.add_fixed(2 * em)

        self.switch_the_pre = gui.ToggleSwitch("Choose the N/pre N component(s)")
        self.switch_the_pre.set_on_clicked(self._on_switch_the_pre)
        pannel.add_child(self.switch_the_pre)
        pannel.add_fixed(0.5 * em)

        # Add two number editors, one for integers and one for floating point
        # Number editor can clamp numbers to a range, although this is more
        # useful for integers than for floating point.
        self.k = gui.NumberEdit(gui.NumberEdit.INT)
        self.k.int_value = 1
        self.k.set_limits(0, 19)  # value coerced to 1
        self.k.set_on_value_changed(self._on_k_change)
        numlayout = gui.Horiz()
        numlayout.add_child(gui.Label("Number of the component"))
        numlayout.add_child(self.k)
        numlayout.add_fixed(em)  # manual spacing (could set it in Horiz() ctor)
        pannel.add_child(numlayout)
        pannel.add_fixed(0.5 * em)

        # Create a slider. It acts very similar to NumberEdit except that the
        # user moves a slider and cannot type the number.
        slicer_layout = gui.Horiz()
        self.v = gui.Slider(gui.Slider.DOUBLE)
        self.v.set_limits(-3, 3)
        self.v.set_on_value_changed(self._on_v_change)
        resetbutton = gui.Button("Reset")
        resetbutton.horizontal_padding_em = 0.5
        resetbutton.vertical_padding_em = 0
        resetbutton.set_on_clicked(self._on_reset_button)
        slicer_layout.add_child(self.v)
        slicer_layout.add_child(resetbutton)
        pannel.add_child(slicer_layout)
        pannel.add_fixed(2 * em)

        # btn_layout = gui.Horiz()
        # fit_btn = gui.Button('fit')
        # fit_btn.set_on_clicked(self._on_fit_clicked)
        # btn_layout.add_stretch()
        # btn_layout.add_child(fit_btn)
        # pannel.add_child(btn_layout)
        # pannel.add_fixed(5 * em)

        self._filechoose = gui.TextEdit()
        fit_btn = gui.Button("fit")
        fit_btn.horizontal_padding_em = 0.5
        fit_btn.vertical_padding_em = 0
        fit_btn.set_on_clicked(self._on_fit_clicked)

        # (Create the horizontal widget for the row. This will make sure the
        # text editor takes up as much space as it can.)
        filechoose_layout = gui.Horiz()
        filechoose_layout.add_child(gui.Label("fit file:"))
        filechoose_layout.add_child(self._filechoose)
        filechoose_layout.add_fixed(0.25 * em)
        filechoose_layout.add_child(fit_btn)
        pannel.add_child(filechoose_layout)
        pannel.add_fixed(0.5 * em)

        return pannel

    def _on_compute_chamfer_button(self):
        if self.mesh is None:
            return
        Y = self.get_pts()
        X = np.array(self.mesh.vertices)

        dis_chamfer = distance_Chamfer(X, Y)
        d1 = directed_hausdorff(X, Y)[0]
        d2 = directed_hausdorff(Y, X)[0]
        dis_hausdorff = np.max([d1, d2])

        self.Chamfer.text_value = f'{dis_chamfer:.6f}'
        self.Hausdorff.text_value = f'{dis_hausdorff:.6f}'

    def _on_layout(self, layout_context):
        #   在on_layout回调函数中应正确设置所有子对象的框架(position + size)，
        #   回调结束之后才会布局孙子对象。
        r = self.window.content_rect

        pannel_width = 17 * layout_context.theme.font_size
        pannel_height = min(
            r.height, self._pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._scene.frame = gui.Rect(r.x, r.y, r.get_right() - pannel_width, r.get_bottom())
        self._pannel.frame = gui.Rect(r.get_right() - pannel_width, r.y, pannel_width, r.get_bottom())

    # open file dir
    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Select directory",
                                 self.window.theme)
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        self.window.close_dialog()

    def _on_filedlg_done(self, path):
        self._fileedit.text_value = path
        self.window.close_dialog()
        self.path = path

        scale = None
        names = None
        if os.path.exists(os.path.join(path, 'scale')):
            f = open(os.path.join(path, 'scale'), "r", encoding="utf-8")
            name_scale = json.load(f)
            names = list(name_scale.keys())
            scale = np.array(list(name_scale.values()))
        try:
            self.X = get_all(path, names)
        except IOError or ValueError:
            print('IOError')
            return
        if self.X is not None:
            self.pca = TraditionalPCA(self.X, scale)
            self.init_scene(self.get_pts())
            self.update_shape()
            # self.reset()

    def reset(self):
        self.total_num.int_value = 0
        self.contribution.text_value = '0.00%'
        self.RMSE.text_value = '0.000mm'
        self.Chamfer.text_value = '0.000000'
        self.Hausdorff.text_value = '0.000000'
        self._on_reset_button()

    # make b change: Y = m + pb
    def _on_switch_the_pre(self, is_on):
        if is_on:
            print("Choose pre")
        else:
            print("Choose one")

    def _on_n_change(self, n):
        if self.pca is not None:
            if 0 < n < self.pca.N:
                counts, distance = self.pca.print_counts(int(n))
                # print(counts, distance)
                self.contribution.text_value = f'{counts*100:.2f}%'
                self.RMSE.text_value = f'{distance:.3f}mm'
            else:
                print('Index is out of boundary.')
        else:
            print('Please choose a directory and load data first.')

    def _on_k_change(self, k):
        self.update_shape()

    def _on_v_change(self, v):
        self.update_shape()

    def _on_reset_button(self):
        self.k.int_value = 1
        self.v.double_value = 0
        self.update_shape()

    def get_pts(self):
        k = self.k.int_value
        v = self.v.double_value
        if self.pca is not None:
            if 0 < k < self.pca.N:
                isPre = self.switch_the_pre.is_on
                if isPre:
                    pts = self.pca.get_pre(k, v).reshape(-1, 3)
                else:
                    pts = self.pca.get_one(k, v).reshape(-1, 3)
                return pts
            else:
                print('Index is out of boundary.')
        else:
            print('Please choose a directory and load data first.')

        return None

    # display form: PointCloud/Mesh
    def _on_switch_type(self, is_on):
        self.update_shape()

    def init_scene(self, pts):
        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.shader = "defaultLit"
        self.mat.point_size = 5

        self.pc.points = o3d.utility.Vector3dVector(pts)
        bounds = self.pc.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def update_shape(self):
        pts = self.get_pts()
        if pts is None:
            return

        self._scene.scene.remove_geometry(BoneModelVisApp.PointCloud)
        self._scene.scene.remove_geometry(BoneModelVisApp.Mesh)
        if self.switch_PointCloud_Mesh.is_on:
            self.mesh = self.get_mesh(pts)
            if self.mesh is not None:
                self._scene.scene.add_geometry(BoneModelVisApp.Mesh, self.mesh, self.mat)
        else:
            self.mesh = None
            self.pc.points = o3d.utility.Vector3dVector(pts)
            self.pc.paint_uniform_color([0, 0, 1])
            self._scene.scene.add_geometry(BoneModelVisApp.PointCloud, self.pc, self.mat)

    def _on_mesh_type_change(self, idx):
        self.update_shape()

    def get_mesh(self, pts):
        mesh_type = self.mesh_type.selected_value
        if mesh_type == 'Static Topology':
            mesh = o3d.io.read_triangle_mesh(os.path.join(self.path, 'example.ply'))
            mesh.vertices = o3d.utility.Vector3dVector(pts)
        elif mesh_type == 'Optimization SAP':
            mesh = get_optimize_mesh(get_target(pts))
        elif mesh_type == 'Learning SAP':
            toNpz(pts)
            mesh = main_generate()
        else:
            return None

        mesh.compute_triangle_normals()
        return mesh

    # others
    def _on_menu_checkable(self):
            gui.Application.instance.menubar.set_checked(
                BoneModelVisApp.MENU_CHECKABLE,
                not gui.Application.instance.menubar.is_checked(
                    BoneModelVisApp.MENU_CHECKABLE))

    def _on_menu_quit(self):
        gui.Application.instance.quit()
        exit(0)

    def _on_fit_clicked(self):
        file_path = ""
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select File",
                                 self.window.theme)
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        def choose(path):
            file_path = path
            self._filechoose.text_value = path
            self.window.close_dialog()

            window = gui.Application.instance.create_window("fit_window", 1000, 600)
            # 渲染窗口
            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(window.renderer)
            window.add_child(scene)

            V = np.loadtxt(path)
            pc1, pc2 = self.pca.fit(V, 3)

            p1 = o3d.geometry.PointCloud()
            p1.points = o3d.utility.Vector3dVector(pc1)
            p1.paint_uniform_color([1, 0, 0])

            p2 = o3d.geometry.PointCloud()
            p2.points = o3d.utility.Vector3dVector(pc2)
            p2.paint_uniform_color([0, 1, 0])

            mix = np.concatenate((pc1, pc2))
            bounds = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(mix))
            scene.setup_camera(60, bounds, bounds.get_center())

            scene.scene.add_geometry("1", p1, self.mat)
            scene.scene.add_geometry("2", p2, self.mat)

        filedlg.set_on_done(choose)
        self.window.show_dialog(filedlg)

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":
    app = BoneModelVisApp()
    app.run()
