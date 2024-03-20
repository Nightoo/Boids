import numpy as np
from scipy.spatial.distance import cdist
from numba import njit, prange

@njit
def mean_axis0(arr):
    """
    Заменяет numpy функцию np.mean(axis=0), чтобы можно было использовать numba
    """
    n = arr.shape[1]
    res = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        res[i] = arr[:, i].mean()
    return res


def init_boids(boids: np.ndarray, asp: float, vrange: tuple = (0., 1.)):
    """
    Инициализация агентов, задание начальных координат и скорости
    """
    n = boids.shape[0]
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2*np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s


@njit
def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """
    Отрисовка стрелок.  Стрелка рисуется во второй паре координат. Направление - от первых координат ко вторым
    """
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))


@njit
def clip_mag(arr: np.ndarray, lims = (0., 1.)):
    """
    Ограничение скорости (чтобы не росла бесконечно)
    """
    v = np.sum(arr * arr, axis=1)**0.5
    mask = v > 0
    v_clip = np.clip(v, *lims)
    arr[mask] *= (v_clip[mask] / v[mask]).reshape(-1, 1)


@njit
def propagate(boids: np.ndarray, dt: float, vrange: tuple):
    """
    Задание движения: v = v + a * dt; x = x + v * dt
    """
    boids[:, 2:4] += dt * boids[:, 4:6]  # к скорости прибавляем dt умножить на ускорение
    clip_mag(boids[:, 2:4], lims=vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]  # к координате прибавляем dt умножить на скорость


@njit
def coloring(agent_index, D, perception, mask_n):
    """
    Задание цвета: красный - выбранный агент, зеленый - видимые, синий - используемые для расчетов
    """
    colors = []
    d = D[agent_index]  # из массива с дистанциями выбираем строчку для выбранного агента
    n = mask_n[agent_index]  # выбираем дистанции для n ближайших соседей
    for i in range(len(d)):
        col = 'black'  # по умолчанию не выделяем цветом
        if (d[i] < perception) and i != agent_index:  # все, кто попад в поле зрения - зеленого цвета
            col = 'green'
        if n[i]:  # если агент среди n ближайших соседей - синего цвета
            col = 'blue'
        if i == agent_index:  # сам выбранный агент красного цвета
            col = 'red'
        colors.append(col)
    return colors

@njit
def walls(boids: np.ndarray, asp: float):
    """
    Избегание стен. Чем ближе к стене, тем больше ускорение от стен.
    Использует функцию 10 / ((|x| + c) ** 40), чтобы агенты могли быть достаточно близко к стене, но прохождение сквозь стену почти невозможно.
    """
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]

    a_left = 10 / (np.abs(x) + c)**40
    a_right = -10 / (np.abs(x - asp) + c)**40

    a_bottom = 10 / (np.abs(y) + c)**40
    a_top = -10 / (np.abs(y - 1.) + c)**40

    return np.column_stack((a_left + a_right, a_bottom + a_top))


@njit
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    """
    Сплоченность. Направляет агента к средней координате соседей.
    """
    center = mean_axis0(boids[neigh_mask, :2])
    a = (center - boids[idx, :2]) / perception
    return a


@njit
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray) -> np.ndarray:
    """
    Разделение. Изгебание столковений с соседями.
    """
    d = mean_axis0(boids[neigh_mask, :2] - boids[idx, :2])
    return -d / ((d[0]**2 + d[1]**2) + 1)


@njit
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:
    """
    Выравнивание. Направление изменяется, чтобы соответствовать направлениям соседей.
    """
    v_mean = mean_axis0(boids[neigh_mask, 2:4])
    a = (v_mean - boids[idx, 2:4]) / (2 * vrange[1])
    return a

@njit
def noise(boids):
    """
    Шум. Случайное изменение направления, символизирующее случайные факторы
    """
    return np.random.uniform(-1, 1, 2)


def distance(boids: np.ndarray) -> np.ndarray:
    """
    Матрица расстояний между агентами.
    """
    return cdist(boids[:, :2], boids[:, :2])


@njit(parallel=True)
def flocking(boids: np.ndarray,
             D: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             vrange:tuple,
             n_neighbors: int):
    """
    Основная функция. Считаем все ускорения на каждой итерации
    """
    N = boids.shape[0]
    mask_n = np.full((N, N), False)
    for i in prange(N):
        D[i, i] = perception + 1  # изменеям растояние агента с самим собой, чтобы не мешало при расчетах

        ind = D[i].argsort()[:n_neighbors] # выбираем n ближайших соседей
        l = np.arange(0, N)
        for j in range(n_neighbors):
            a = np.where(l == ind[j], True, False)
            mask_n[i] += a

    mask_perception = D < perception  # маска - смотрим соседей, где расстояние между боидами меньше perception
    mask = np.logical_and(mask_n, mask_perception) #нам нужны только те, которые видимы и входят в число n ближайших
    wal = walls(boids, asp)  # стены
    for i in prange(N):
        noi = noise(boids)  # шум
        if not np.any(mask[i]):  # если нет подходящих соседей, учитываются только шум и стены
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)  # сплоченность
            alg = alignment(boids, i, mask[i], vrange)  # разделение
            sep = separation(boids, i, mask[i])  # выравнивание
        a = coeffs[0] * coh + coeffs[1] * alg + \
            coeffs[2] * sep + coeffs[3] * wal[i] + \
            coeffs[4] * noi
        boids[i, 4:6] = a
    return mask  # возвращаем маску соседей, она нужна для выбора цвета
