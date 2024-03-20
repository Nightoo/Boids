from vispy import app, scene
from vispy.geometry import Rect
from vispy.scene.visuals import Text
import numpy as np
from funcs import (
    init_boids, directions, propagate, flocking, distance, coloring
)
# параметры экрана
w, h = 1900, 1060
asp = w / h
# начальные условия
N = 5000
dt = 0.01
perception = 1/10
n_neighbors = 5  # количество соседей, учитывающихся для расчета поведения агентов
vrange=(0.05, 0.1)
#                    c      a    s      w     n
coeffs = np.array([0.02,   0.9,  1.2,  0.15, 0.5])
# создаем агентов
boids = np.zeros((N, 6), dtype=np.float64)  # для каждого боида 6 значений - координата, скорость и ускорения для x и y
init_boids(boids, asp, vrange=vrange)

# выбор агента, для которого бует выполнено отображение
agent_index = 0
agent = boids[agent_index]
# начальные цвета
colors = ['white' for _ in range(N)]
# размер агентов
arrow_size = 3
# создание экрана vispy
canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))
# задание цветов и поля видимости
arrow_colors = scene.Markers(pos=boids[:, :2], size=arrow_size, edge_color=colors, face_color=colors, parent=view.scene)
marker = scene.Markers(pos=np.array([[agent[0], agent[1]]]), size=2 * h * perception, alpha=0, edge_color='red', face_color='black', parent=view.scene)
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=colors,
                     # width=5,
                     arrow_size=arrow_size,
                     connect='segments',
                     parent=view.scene)

# параметры текста
t_n = Text('N = ' + str(N), font_size=8, parent=view.scene, color='green')
t_n.pos = 0.1, 0.175
t_c = Text('cohesion = ' + str(coeffs[0]), font_size=8, parent=view.scene, color='green')
t_c.pos = 0.1, 0.15
t_a = Text('alignment = ' + str(coeffs[1]), font_size=8, parent=view.scene, color='green')
t_a.pos = 0.1, 0.125
t_s = Text('separation = ' + str(coeffs[2]), font_size=8, parent=view.scene, color='green')
t_s.pos = 0.1, 0.1
t_w = Text('walls = ' + str(coeffs[3]), font_size=8, parent=view.scene, color='green')
t_w.pos = 0.1, 0.075
t_noi = Text('noise = ' + str(coeffs[4]), font_size=8, parent=view.scene, color='green')
t_noi.pos = 0.1, 0.05
t_fps = Text(str(0), font_size=8, parent=view.scene, color='green')
t_fps.pos = 0.1, 0.95


def update(event):
    D = distance(boids)  # считаем попарное расстояние между всеми боидами, вынесено в отдельную функцию, так как cdist работает быстрее написанной функции с @njit
    mask_neighbors = flocking(boids, D, perception, coeffs, asp, vrange, n_neighbors)  # рассчитываем ускорение на следующий шаг
    propagate(boids, dt, vrange)  # рассчитываем движение
    col = coloring(agent_index, D, perception, mask_neighbors)  # рассчитываем цвета
    marker.set_data(pos=np.array([[agent[0], agent[1]]]), size=2 * h * perception, edge_color='red', face_color='black')  # обновляем отображение видимости
    arrow_colors.set_data(pos=boids[:, :2], size=arrow_size, edge_color=col, face_color=col)  # обновляем цвета
    arrows.set_data(arrows=directions(boids, dt))  # обновляем положение
    t_fps.text = str(round(canvas.fps)) + ' FPS'  # обновляем фпс
    canvas.update()


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()