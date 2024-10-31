import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSlider, QPushButton,
                             QSizePolicy, QLabel, QLineEdit)
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl.items import GLTextItem
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
from time import perf_counter, sleep
import signal

red = np.array([0.7, 0, 0, 1])
green = np.array([0, 0.7, 0, 1])
blue = np.array([0, 0, 0.7, 1])
dark_red = np.array([0.3, 0, 0, 1])
dark_green = np.array([0, 0.3, 0, 1])
dark_blue = np.array([0, 0, 0.3, 1])
white = np.array([1, 1, 1, 1])
grey = np.array([0.3, 0.3, 0.3, 1])
yellow = np.array([223.0/255.0, 238.0/255.0, 95.0/255.0, 1.0])

class TransformMPL:

    def __init__(self, A, ax=None):
        self.A = A

    def update(self, A):
        self.A = A

    def show(self):
        plt.show()


class PlanarMPL:

    def __init__(self, arm, q0: None|list=None, trace: bool=False, ax: None|plt.Axes=None):
        self.arm = arm
        if ax is not None:
            fig = ax.figure
            flag = True
        else:
            fig, ax = plt.subplots()
            flag = False
        self.fig: plt.Figure = fig
        self.ax: plt.Axes = ax
        self.n = arm.n
        self.reach = arm.reach
        if not flag:
            plt.axis('equal')
            plt.xlim([-arm.reach * 1.5, arm.reach * 1.5])
            plt.ylim([-arm.reach * 1.5, arm.reach * 1.5])

        self.joints: list[Circle] = []
        self.links: list[plt.Line2D] = []
        self.joints.append(self.ax.add_patch(Circle([0, 0], 0.15, color=[0,0,0,1])))

        if q0 is None:
            q0 = [0] * self.n
        self.q0 = q0
        for i in range(self.n):
            A = self.arm.fk(q0, index=i)
            A_next = self.arm.fk(q0, index=i+1)

            if i != 0:
                joint = Circle(A[0:2, 3], 0.1, color=[0,0,0,1])
                self.joints.append(self.ax.add_patch(joint))
            link, = self.ax.plot([A[0,3], A_next[0,3]], [A[1,3], A_next[1,3]], lw=3, color=[0,0,0,1])
            self.links.append(link)

        end_effector = Circle(A_next[0:2,3], 0.1, color=[0.5,0,0,1])
        self.joints.append(self.ax.add_patch(end_effector))

        self.do_trace = trace
        if trace:
            self.start_trace(q0)

    def start_trace(self, q0):
        self.xs: list[list[float]] = []
        self.ys: list[list[float]] = []
        self.traces: list[plt.Line2D] = []
        for i in range(self.n):
            pos = self.arm.fk(q0, index=i + 1)[0:2, 3]
            C = [0, 0, 0, 0.4]
            C[i] = 0.5
            line, = self.ax.plot(pos[0], pos[1], lw=1, color=C)
            self.xs.append([pos[0]])
            self.ys.append([pos[1]])
            self.traces.append(line)

    def update(self, q):

        for i in range(self.n):
            A = self.arm.fk(q, index=i)
            A_next = self.arm.fk(q, index=i+1)

            if i != 0:
                self.joints[i].set_center(A[0:2, 3])

            self.links[i].set_xdata([A[0,3], A_next[0,3]])
            self.links[i].set_ydata([A[1,3], A_next[1,3]])
        self.joints[-1].set_center(A_next[0:2, 3])

        if self.do_trace:
            for i in range(self.n):
                pos = self.arm.fk(q, index=i+1)[0:2,3]
                self.xs[i].append(pos[0])
                self.ys[i].append(pos[1])
                self.traces[i].set_xdata(self.xs[i])
                self.traces[i].set_ydata(self.ys[i])

        plt.pause(0.02)

    def show(self):
        plt.show()

    def set_bounds(self, xbound=None, ybound=None):
        print('finish me')

    def play(self):
        # clear all the line plots
        if self.do_trace:
            self.do_trace = False
            for i in range(self.n):
                self.ax.lines.remove(self.traces[i])
                # NOTE: This might be what you want if line above doesn't work - Mat
                # self.ax.remove(self.traces[i])

        # move to the default position
        self.update(self.q0)

        # resize and create slider bars
        self.ax.set_position([0, 0, 0.75, 1])
        # NOTE: I think these changes work and isolate the plot. If not, revert to the original code - Mat
        self.ax.axis('equal')
        self.ax.set_xlim([-self.arm.reach * 1.5, self.arm.reach * 1.5])
        self.ax.set_ylim([-self.arm.reach * 1.5, self.arm.reach * 1.5])
        # plt.axis('equal')
        # plt.xlim([-self.arm.reach * 1.5, self.arm.reach * 1.5])
        # plt.ylim([-self.arm.reach * 1.5, self.arm.reach * 1.5])

        max_h = 0.2
        min_h = 0.8 / self.n
        h = min(max_h, min_h)
        self.sliders: list[Slider] = []
        self.gui_axes: list[plt.Axes] = []

        def get_text_from_A(A):
            pos = np.around(A[0:2, 3], decimals=2)
            theta = np.around(np.arctan2(A[1, 0], A[0, 0]), decimals=2)

            s = 'Pos: [' + str(pos[0]) + ', ' + str(pos[1]) + ']\n'
            s += 'Angle: [' + str(theta) + ']\n'

            return s
        text_pos = [-self.reach * 1.35, self.reach * 1.15]
        self.text_box = self.ax.text(text_pos[0], text_pos[1], get_text_from_A(self.arm.fk(self.q0)))

        def slider_update(val):
            q = self.q0
            for i in range(self.n):
                q[i] = self.sliders[i].val
            self.update(q)
            self.text_box.set_text(get_text_from_A(self.arm.fk(q)))
            plt.draw()

        for i in range(self.n):
            self.gui_axes.append(self.fig.add_subplot())
            self.gui_axes[i].set_position([0.775, 0.8 - h * i, 0.15, h - 0.05])
            self.sliders.append(Slider(ax=self.gui_axes[i],
                                        label=str(i),
                                        valmin=-np.pi,
                                        valmax=np.pi,
                                        valinit=0,
                                        orientation='horizontal'))
            self.sliders[i].on_changed(slider_update)

        # plt.show()


class VizScene:
    """The viz scene holds all the 3d objects to be plotted. This includes arms (which are GLMeshObjects), transforms
    (which are plots or quiver type things), scatter points, and lines."""
    def __init__(self):
        self.arms: list[ArmMeshObject] = []
        self.frames: list[FrameViz] = []
        self.axes: list[AxisViz] = []
        self.markers: list[gl.GLMeshItem] = []
        self.obstacles: list[gl.GLMeshItem] = []
        self.range = 5

        if QApplication.instance() is None:
            self.app = pg.QtWidgets.QApplication([])
        else:
            self.app = QApplication.instance()
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('Robot Visualization 2: The Sequel')
        self.window.setGeometry(200, 100, 1200, 900)
        self.grid = gl.GLGridItem(color=(0, 0, 0, 76.5))
        self.grid.scale(1, 1, 1)
        self.window.addItem(self.grid)
        self.window.setCameraPosition(distance=self.range)
        self.window.setBackgroundColor('w')
        self.window.show()
        self.window.raise_()
        self.window.opts['center'] = pg.Vector(0, 0, 0)

        self.app.processEvents()

    def add_arm(self, arm, draw_frames=False, label_axes=False, joint_colors=None):
        a = ArmMeshObject(arm, draw_frames=draw_frames, label_axes=label_axes, joint_colors=joint_colors)
        self.arms.append(a)
        a.update()
        self.window.addItem(a.mesh_object)
        if draw_frames:
            for frame in a.frame_objects:
                for axis in frame.axes:
                    self.window.addItem(axis)
                if label_axes:
                    for txt in frame.axis_labels:
                        self.window.addItem(txt)

        if 2 * arm.reach > self.range:
            self.range = 2 * arm.reach
            self.window.setCameraPosition(distance=self.range)

        self.app.processEvents()

    def remove_arm(self, arm=None):
        if arm is None:
            for arm in self.arms:
                self.window.removeItem(arm.mesh_object)
                if arm.draw_frames:
                    for frame in arm.frame_objects:
                        for axis in frame.axes:
                            self.window.removeItem(axis)
                        if arm.label_axes:
                            for txt in frame.axis_labels:
                                self.window.removeItem(txt)
            self.arms.clear()
        elif isinstance(arm, (int)):
            self.window.removeItem(self.arms[arm].mesh_object)
            if self.arms[arm].draw_frames:
                for frame in self.arms[arm].frame_objects:
                    for axis in frame.axes:
                        self.window.removeItem(axis)
                    if self.arms[arm].label_axes:
                        for txt in frame.axis_labels:
                            self.window.removeItem(txt)
            self.arms.pop(arm)
        else:
            print("Warning: invalid index entered!")
            return None
        self.app.processEvents()

    def add_frame(self, A, label=None, axes_label=None):
        self.frames.append(FrameViz(A, frame_label=label, axes_label=axes_label))
        self.window.addItem(self.frames[-1].axes[0])
        self.window.addItem(self.frames[-1].axes[1])
        self.window.addItem(self.frames[-1].axes[2])
        if label is not None:
            self.window.addItem(self.frames[-1].frame_label)
        if axes_label is not None:
            for txt in self.frames[-1].axis_labels:
                self.window.addItem(txt)

            # this was from the old version of text rendering. I don't think it's needed, but
            # seems to assign a windwo to render to, so I don't know for sure. - Killpack
            # self.frames[-1].label.setGLViewWidget(self.window)

        if 2 * np.linalg.norm(A[:3,3]) > self.range:
            self.range = 2 * np.linalg.norm(A[:3,3])
            self.window.setCameraPosition(distance=self.range)

        self.app.processEvents()

    def remove_frame(self, ind=None):
        if ind is None:
            for frame in self.frames:
                for axis in frame.axes:
                    self.window.removeItem(axis)
                if frame.frame_label is not None:
                    self.window.removeItem(frame.frame_label)
                if frame.axis_labels is not None:
                    for txt in frame.axis_labels:
                        self.window.removeItem(txt)
            self.frames.clear()
        elif isinstance(ind, (int)):
            for axis in self.frames[ind].axes:
                self.window.removeItem(axis)
            if self.frames[ind].frame_label is not None:
                self.window.removeItem(self.frames[ind].frame_label)
            if self.frames[ind].axis_labels is not None:
                for txt in self.frames[ind].axis_labels:
                    self.window.removeItem(txt)
            self.frames.pop(ind)
        else:
            print("Warning: invalid index entered!")
            return None
        self.app.processEvents()

    def add_axis(self, axis, pos_offset=np.zeros(3), label=None):
        self.axes.append(AxisViz(axis, pos_offset, label=label))
        self.window.addItem(self.axes[-1].axis)
        if label is not None:
            self.window.addItem(self.axes[-1].label)

        self.app.processEvents()

    def remove_axis(self, ind=None):
        if ind is None:
            for axis in self.axes:
                self.window.removeItem(axis.axis)
                if axis.label is not None:
                    self.window.removeItem(axis.label)
            self.axes.clear()
        elif isinstance(ind, (int)):
            ax_viz = self.axes[ind]
            self.window.removeItem(ax_viz.axis)
            if ax_viz.label is not None:
                self.window.removeItem(ax_viz.label)
            self.axes.pop(ind)
        else:
            print("Warning: invalid index entered!")
            return None
        self.app.processEvents()

    def add_marker(self, pos, color=green, radius=0.1):
        if not isinstance(pos, (np.ndarray)):
            pos = np.array(pos)

        marker = gl.MeshData.sphere(rows=20, cols=20, radius=radius)

        mesh_marker = gl.GLMeshItem(
            meshdata=marker,
            smooth=True,
            color=color
        )
        mesh_marker.translate(*pos)

        self.markers.append(mesh_marker)
        self.window.addItem(self.markers[-1])

        if 2 * np.linalg.norm(pos) > self.range:
            self.range = 2 * np.linalg.norm(pos)
            self.window.setCameraPosition(distance=self.range)

        self.app.processEvents()

    def remove_marker(self, ind=None):
        if ind is None:
            for marker in self.markers:
                self.window.removeItem(marker)
            self.markers = []
        elif isinstance(ind, (int)):
            self.window.removeItem(self.markers[ind])
            self.markers.pop(ind)
        else:
            print("Warning: invalid index entered!")
            return None
        self.app.processEvents()


    def add_obstacle(self, pos, color=yellow, rad = 1.0):
        if not isinstance(pos, (np.ndarray)):
            pos = np.array(pos)

        mobst = gl.MeshData.sphere(rows=20, cols=20, radius=rad)

        m1 = gl.GLMeshItem(
            meshdata=mobst,
            smooth=True,
            color=yellow
        )
        m1.translate(*pos)

        self.obstacles.append(m1)
        self.window.addItem(self.obstacles[-1])

        self.app.processEvents()

    def remove_obstacle(self, ind=None):
        if ind is None:
            for obstacle in self.obstacles:
                self.window.removeItem(obstacle)
            self.obstacles = []
        elif isinstance(ind, (int)):
            self.window.removeItem(self.obstacles[ind])
            self.obstacles.pop(ind)
        else:
            print("Warning: invalid index entered!")
            return None
        self.app.processEvents()

    def update(self, qs=None, As=None, poss=None):
        if qs is not None:
            if isinstance(qs[0], (list, tuple, np.ndarray)):
                for i in range(len(self.arms)):
                    self.arms[i].update(qs[i])
            else:
                self.arms[0].update(qs)

        if As is not None:
            if isinstance(As, (list, tuple)):
                for i in range(len(self.frames)):
                    self.frames[i].update(As[i])
            elif len(As.shape) == 3:
                for i in range(len(self.frames)):
                    self.frames[i].update(As[i])
            else:
                self.frames[0].update(As)

        if poss is not None:
            if isinstance(poss, (list, tuple)):
                for i in range(len(self.markers)):
                    if not isinstance(poss[i], (np.ndarray)):
                        pos = np.array(poss[i])
                    else:
                        pos = poss[i]
                    self.markers[i].resetTransform()
                    self.markers[i].translate(*pos)
            else:
                if not isinstance(poss, (np.ndarray)):
                    pos = np.array(poss)
                else:
                    pos = poss
                self.markers[0].resetTransform()
                self.markers[0].translate(*pos)

        self.app.processEvents()
        sleep(0.00001)

    def hold(self):
        while self.window.isVisible():
            self.app.processEvents()

    def wander(self, index=None, q0=None, speed=1e-1, duration=np.inf, accel=5e-4):
        if index is None:
            index = range(len(self.arms))

        tstart = perf_counter()
        t = tstart
        flag = True
        qs = []
        dqs = []

        while t < tstart + duration and self.window.isVisible():
            for i, ind in enumerate(index):
                n = self.arms[ind].n
                if flag:
                    if q0 is None:
                        qs.append(np.zeros((n,)))
                    else:
                        qs.append(q0[i])
                    # dqs.append(np.random.random_sample((n,)) * speed - speed / 2)
                    dqs.append(np.zeros((n,)))

                dqq = np.zeros((n,))
                for j in range(n):
                    s = dqs[i][j] / speed
                    dqq[j] = dqq[j] + np.random.random_sample((1,)) * accel - accel / 2 - accel * s**3
                dqs[i] = dqs[i] + dqq
                qs[i] = qs[i] + dqs[i]
                self.arms[ind].update(qs[i])

            if flag:
                flag = False
            t = perf_counter()
            self.app.processEvents()

    def close_viz(self):
        self.app.closeAllWindows()


class ArmPlayer:
    def __init__(self, arm, fontsize=14):
        if QApplication.instance() is None:
            self.app = pg.QtWidgets.QApplication([])
        else:
            self.app = QApplication.instance()
        self.window = QMainWindow()
        font = self.window.font()
        font.setPointSize(fontsize)
        self.window.setFont(font)
        self.window.setGeometry(200, 300, 1000, 700)
        self.window.setWindowTitle("Arm Play")

        self.main_layout = QHBoxLayout()
        w1 = gl.GLViewWidget()
        w1.setBackgroundColor('w')
        grid = gl.GLGridItem(color=(0, 0, 0, 76.5))
        grid.scale(1, 1, 1)
        w1.addItem(grid)
        self.arm = ArmMeshObject(arm)
        self.n = arm.n
        self.jt = arm.jt
        r_scale = 2 # slider increments by 1/2 for revelute joints
        r_max = 180
        p_scale = 100 # slider increments by 1/100 for prismatic joints
        p_max = 1.
        self.jt_scale = np.array([r_scale if jt == 'r' else p_scale for jt in arm.jt])
        if arm.qlim is None:
            self.jt_lims = np.array([
                [-r_max if jt == 'r' else -p_max for jt in arm.jt],
                [r_max if jt == 'r' else p_max for jt in arm.jt]
                ], dtype=np.float64).T
        else:
            # expecting qlim for 'r' joints to be in radians
            self.jt_lims = np.array(arm.qlim, dtype=np.float64).T
            mask = [True if jt == 'r' else False for jt in arm.jt]
            self.jt_lims[mask] = np.rad2deg(self.jt_lims[mask])
            assert self.jt_lims.shape == (self.n,2)

        w1.addItem(self.arm.mesh_object)
        w1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        w2 = QVBoxLayout()
        self.sliders: list[QSlider] = []
        self.slider_textboxes: list[QLineEdit] = []
        for i in range(arm.n):
            line = QHBoxLayout()
            t1 = QLabel()
            t1.setText(f"Joint {i + 1}: ")
            t1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            box = QLineEdit()
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            box.setText("0.0")
            box.editingFinished.connect(self.update_textboxes)
            t2 = QLabel()
            if arm.jt[i] == 'r':
                t2.setText("(degrees)")
            else:
                t2.setText("(meters)")
            t2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            line.addWidget(t1)
            line.addWidget(box)
            line.addWidget(t2)
            s = QSlider(Qt.Horizontal)
            s.setRange(*(self.jt_lims[i] * self.jt_scale[i]).astype(int))
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            s.valueChanged.connect(self.update_sliders)
            self.sliders.append(s)
            self.slider_textboxes.append(box)
            w2.addLayout(line, stretch=0)
            w2.addWidget(s, stretch=0)
        button = QPushButton()
        button.setText("Randomize")
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button.pressed.connect(self.button_pressed)
        self.random_button = button
        w2.addWidget(button)
        self.main_layout.addWidget(w1, stretch=3)
        self.main_layout.addLayout(w2, stretch=1)

        w = QWidget()
        w.setLayout(self.main_layout)
        self.window.setCentralWidget(w)
        self.window.show()
        self.window.raise_()

        self.app.processEvents()

        # kill with ctrl-c
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.app.exec()

    # NOTE: The following 3 update functions are inefficient because all buttons
    # and sliders are updated when any one of them are changed. Could be optimized
    # by adding unique update functions for each item in the constructor above.
    # - Mat
    def update_sliders(self):
        qs = np.zeros((self.n,))
        for i, s in enumerate(self.sliders):
            q = s.value() / self.jt_scale[i]
            if self.jt[i] == 'r':
                qs[i] = np.deg2rad(q)
            else:
                qs[i] = q
            self.slider_textboxes[i].setText(f'{q}')
        self.arm.update(qs)

    def update_textboxes(self):
        qs = np.zeros((self.n,))
        for i, b in enumerate(self.slider_textboxes):
            try: # if the text box is empty or not a number, it will throw an error
                q = float(b.text())
                if q > self.jt_lims[i,1]:
                    q = self.jt_lims[i,1]
                elif q < self.jt_lims[i,0]:
                    q = self.jt_lims[i,0]
            except: # if invalid text then just use the current slider value
                q = self.sliders[i].value() / self.jt_scale[i]
            self.sliders[i].setValue(int(q*self.jt_scale[i]))
            if self.jt[i] == 'r':
                q = np.deg2rad(q)
            qs[i] = q
        self.arm.update(qs)

    def button_pressed(self):
        qs = np.empty(self.n)
        for i in range(self.n):
            q = np.random.uniform(*self.jt_lims[i])
            if self.jt[i] == 'r':
                qs[i] = np.radians(q)
            else:
                qs[i] = q
            self.sliders[i].setValue(int(q*self.jt_scale[i]))
        self.update_textboxes()


class ArmMeshObject:
    def __init__(self, arm, link_colors=None, joint_colors=None, q0=None, draw_frames=False, label_axes=False):
        self.arm = arm
        self.n = arm.n
        self.dh = arm.dh
        self.draw_frames = draw_frames
        self.label_axes = label_axes

        if q0 is None:
            q0 = np.zeros((self.n,))
        self.q0 = q0

        self.link_objects = []
        self.frame_objects = []

        if link_colors is None:
            link_colors = [blue] * self.n

        if joint_colors is None:
            joint_colors = [red if jt == 'r' else green for jt in arm.jt]

        dh_array = np.array(self.dh)
        arm_scale = np.max([np.max(dh_array[0:,1:3]), 0.3])
        frame_scale = 1.5
        ee_scale = 2.0

        self.frame_objects.append(FrameViz(scale=arm_scale*frame_scale, axes_label='b'))

        link_width = np.max([arm_scale, 0.10])*0.20
        joint_width = np.max([arm_scale, 0.10])*0.30
        joint_height = np.max([arm_scale, 0.10])*0.70

        # TODO: change rotary joints from cuboids to cylinders.
        # See example from "add_marker" about to easily generate a
        # cylinder mesh. - Killpack

        for i in range(self.n):
            self.link_objects.append(LinkMeshObject(self.dh[i],
                                                    link_width=link_width,
                                                    joint_width=joint_width,
                                                    joint_height=joint_height,
                                                    link_color=link_colors[i],
                                                    joint_color=joint_colors[i]))
            self.frame_objects.append(FrameViz(scale=arm_scale*frame_scale, axes_label=f'{i}'))


        self.ee_object = EEMeshObject(scale=arm_scale*ee_scale)
        self.frame_objects.append(FrameViz(scale=arm_scale*frame_scale, axes_label='t'))

        self.mesh = np.zeros((0, 3, 3))
        self.colors = np.zeros((0, 3, 4))

        self.set_mesh(q0)

        self.mesh_object = gl.GLMeshItem(vertexes=self.mesh,
                                         vertexColors=self.colors,
                                         drawEdges=True,
                                         computeNormals=False,
                                         edgeColor=np.array([0, 0, 0, 1]))

    def update(self, q=None):
        # it's highly possible that this could be done by using .translate and .rotate functions
        # to simplify the complexity of recalculating every vertex ever time. But "translate" and
        # "rotate" may both be relative motion (meaning you have to use resetTransform too)...
        # - Killpack
        # Mat: translate() and rotate() are relative to the global origin, so you
        # would need to use resetTransform, do rotate() first, then translate()
        if q is None:
            q = self.q0
        self.set_mesh(q)
        self.mesh_object.setMeshData(vertexes=self.mesh,
                                    vertexColors=self.colors)

    def set_mesh(self, q):
        meshes = []
        colors = []
        if self.draw_frames:
            A = self.arm.fk(q, 0, base=True, tip=False)
            R = A[:3,:3]
            p = A[:3,3]
            self.frame_objects[0].update(A)
        for i in range(self.n):
            A = self.arm.fk(q, i+1, base=True, tip=False)
            R = A[:3,:3]
            p = A[:3,3]
            meshes.append(self.link_objects[i].get_mesh(R, p))
            colors.append(self.link_objects[i].get_colors())
            if self.draw_frames:
                self.frame_objects[i+1].update(A)
        A = self.arm.fk(q, i+1, base=True, tip=True)
        R = A[:3,:3]
        p = A[:3,3]
        if self.draw_frames:
            self.frame_objects[-1].update(A)

        meshes.append(self.ee_object.get_mesh(R, p))
        colors.append(self.ee_object.get_colors())

        self.mesh = np.vstack(meshes)
        self.colors = np.vstack(colors)


class LinkMeshObject:
    def __init__(self, dh, jt='r',
                 link_width=0.1,
                 joint_width=0.15,
                 joint_height=0.25,
                 link_color=None,
                 joint_color=None):

        theta = dh[0]
        d = dh[1]
        a = dh[2]
        alpha = dh[3]

        lw = link_width

        w = joint_width
        h = joint_height

        le = np.sqrt(d**2 + a**2)

        self.link_points = np.array([[0, lw/2, 0],
                                     [0, 0, lw/2],
                                     [0, -lw/2, 0],
                                     [0, 0, -lw/2],
                                     [-le, lw / 2, 0],
                                     [-le, 0, lw / 2],
                                     [-le, -lw / 2, 0],
                                     [-le, 0, -lw / 2]])

        v1 = np.array([-le, 0, 0])
        v2 = np.array([-a, d * np.sin(-alpha), -d * np.cos(-alpha)])

        axis = np.cross(v1, v2)
        n_axis = np.linalg.norm(axis)
        if np.abs(n_axis) < 1e-12:
            R = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            ang = np.arccos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            v = axis / np.linalg.norm(axis)
            V = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            R = np.eye(3) + np.sin(ang) * V + (1 - np.cos(ang)) * V @ V

        self.link_points = self.link_points @ R.T

        self.joint_points = np.array([[0.5 * w, -0.5 * w, -0.5 * h],
                                [-0.5 * w, -0.5 * w, -0.5 * h],
                                [-0.5 * w, -0.5 * w, 0.5 * h],
                                [0.5 * w, -0.5 * w, 0.5 * h],
                                [0.5 * w, 0.5 * w, -0.5 * h],
                                [-0.5 * w, 0.5 * w, -0.5 * h],
                                [-0.5 * w, 0.5 * w, 0.5 * h],
                                [0.5 * w, 0.5 * w, 0.5 * h]])

        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]])

        self.joint_points = self.joint_points @ Rz @ Rx + v2

        if link_color is None:
            link_color = np.array([0, 0, 0.35, 1])
        elif not isinstance(link_color, (np.ndarray)):
            link_color = np.array(link_color)
        self.link_colors = np.zeros((12, 3, 4)) + link_color

        if joint_color is None:
            joint_color = np.array([0.35, 0, 0., 1])
        elif not isinstance(joint_color, (np.ndarray)):
            joint_color = np.array(joint_color)
        self.joint_colors = np.zeros((12, 3, 4)) + joint_color
        self.joint_colors[6:8, :, :] = np.zeros((2, 3, 4)) + grey

    @staticmethod
    def points_to_mesh(link_points, joint_points):

        link_mesh = np.array([[link_points[0], link_points[4], link_points[5]],
                              [link_points[0], link_points[1], link_points[5]],
                              [link_points[1], link_points[5], link_points[6]],
                              [link_points[1], link_points[2], link_points[6]],
                              [link_points[2], link_points[6], link_points[7]],
                              [link_points[2], link_points[7], link_points[3]],
                              [link_points[3], link_points[7], link_points[4]],
                              [link_points[3], link_points[0], link_points[4]],
                              [link_points[0], link_points[2], link_points[3]],
                              [link_points[0], link_points[1], link_points[2]],
                              [link_points[5], link_points[4], link_points[7]],
                              [link_points[5], link_points[6], link_points[7]]])

        joint_mesh = np.array([[joint_points[0], joint_points[1], joint_points[2]],
                         [joint_points[0], joint_points[2], joint_points[3]],
                         [joint_points[0], joint_points[3], joint_points[4]],
                         [joint_points[3], joint_points[4], joint_points[7]],
                         [joint_points[2], joint_points[3], joint_points[6]],
                         [joint_points[3], joint_points[6], joint_points[7]],
                         [joint_points[0], joint_points[1], joint_points[5]],
                         [joint_points[0], joint_points[4], joint_points[5]],
                         [joint_points[1], joint_points[2], joint_points[6]],
                         [joint_points[1], joint_points[5], joint_points[6]],
                         [joint_points[4], joint_points[5], joint_points[6]],
                         [joint_points[4], joint_points[6], joint_points[7]]])

        return np.vstack((link_mesh, joint_mesh))

    def get_colors(self):
        return np.vstack((self.link_colors, self.joint_colors))

    def get_mesh(self, R, p):
        lp = self.link_points @ R.T + p
        jp = self.joint_points @ R.T + p

        return self.points_to_mesh(lp, jp)


class FrameViz:
    # NOTE: This class could likely be changed to use the AxisViz class to reduce
    # code duplication. - Mat

    def __init__(self, A=np.eye(4), scale=1, colors=[red,green,blue], frame_label=None, axes_label=None):
        height = 0.35 * scale
        radius = height / 20
        self.frame_label_pos = np.array([0, 0, -0.05])
        self.axis_label_poss = np.eye(3) * height
        # gives mesh for cylinder along positive z-axis starting at origin
        cylinder_mesh = gl.MeshData.cylinder(rows=10, cols=20, radius=[radius]*2, length=height)
        self.faces = cylinder_mesh.faces()
        z_pts = cylinder_mesh.vertexes().copy()
        y_pts = z_pts @ np.array([[1,0,0],[0,0,-1],[0,1,0]])
        x_pts = z_pts @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
        self.pts = [x_pts, y_pts, z_pts]
        self.axes = [gl.GLMeshItem(meshdata=cylinder_mesh, smooth=True, color=colors[i]) for i in range(3)]
        self.frame_label = frame_label
        if self.frame_label is not None:
            self.frame_label = GLTextItem.GLTextItem(text=frame_label, color=(0,0,0))
        self.axis_labels = axes_label
        if self.axis_labels is not None:
            self.axis_labels = [
                GLTextItem.GLTextItem(text=f'x_{axes_label}', color=(0,0,0)),
                GLTextItem.GLTextItem(text=f'y_{axes_label}', color=(0,0,0)),
                GLTextItem.GLTextItem(text=f'z_{axes_label}', color=(0,0,0)),
            ]
        self.update(A)


    def update(self, A: np.ndarray):
        for pts, axis in zip(self.pts, self.axes):
            axis.setMeshData(vertexes=pts @ A[:3,:3].T + A[:3,3], faces=self.faces)
        if self.frame_label is not None:
            p = A[:3,3] + A[:3,:3] @ self.frame_label_pos
            self.frame_label.setData(pos=p)
        if self.axis_labels is not None:
            for pos, label in zip(self.axis_label_poss, self.axis_labels):
                p = A[:3,3] + A[:3,:3] @ pos
                label.setData(pos=p)


class AxisViz:

    def __init__(self, axis: np.ndarray, pos_offset=np.zeros(3), scale=1, color=np.zeros(4), label=None):
        height = 0.35 * scale
        radius = height / 20

        axis = np.array(axis, dtype=np.float64)
        assert axis.shape == (3,)
        axis /= np.linalg.norm(axis) # ensure unit vector
        z = np.array([0,0,1.])
        if np.equal(z, axis).all():
            R = np.eye(3)
        elif np.equal(-z,axis).all():
            R = np.array([[1.,0,0],[0,-1,0],[0,0,-1]])
        else:
            v = np.cross(z, axis)
            c = z @ axis
            skew_v = np.array([[0,-v[2],v[1]], [v[2],0,-v[0]], [-v[1],v[0],0]])
            R = np.eye(3) + skew_v + (skew_v @ skew_v) / (1 + c)

        # gives mesh for cylinder along positive z-axis starting at origin
        cylinder_mesh = gl.MeshData.cylinder(rows=10, cols=20, radius=[radius]*2, length=height)
        pts = cylinder_mesh.vertexes()
        cylinder_mesh.setVertexes(pts @ R.T + pos_offset)
        self.axis = gl.GLMeshItem(meshdata=cylinder_mesh, smooth=True, color=color)
        self.label = label
        if label is not None:
            self.label = GLTextItem.GLTextItem(pos=height*R[:,2] + pos_offset, text=label, color=(0,0,0))


class EEMeshObject:
    def __init__(self, scale=1, w=0.05, o1=0.05, o2=0.15, o3=0.3, o4=0.2, o5=0.1):
        # TODO: can replace the triangle with a fun STL
        # see - https://stackoverflow.com/questions/71052955/is-there-any-way-to-insert-a-3d-image-of-stl-obj-fbx-and-dae-formats-in-pyq

        w = w * scale
        o1 = o1 * scale
        o2 = o2 * scale
        o3 = o3 * scale
        o4 = o4 * scale
        o5 = o5 * scale

        self.points = np.array([[-o1, o2, w/2],
                                [-o1, -o2, w/2],
                                [o3, 0, w/2],
                                [-o1, o2, -w/2],
                                [-o1, -o2, -w/2],
                                [o3, 0, -w/2]
                                ])
        self.colors = np.zeros((8,3,4)) + red
        self.colors[1,:,:] = np.zeros((3,4)) + dark_blue

    @staticmethod
    def points_to_mesh(p):
        mesh = np.array([[p[0], p[1], p[2]],
                         [p[3], p[4], p[5]],
                         [p[0], p[1], p[3]],
                         [p[1], p[4], p[3]],
                         [p[1], p[2], p[5]],
                         [p[1], p[4], p[5]],
                         [p[0], p[2], p[5]],
                         [p[0], p[3], p[5]]
                         ])
        return mesh

    def get_mesh(self, R, p):
        points = self.points @ R.T + p
        mesh = self.points_to_mesh(points)
        return mesh

    def get_colors(self):
        return self.colors
