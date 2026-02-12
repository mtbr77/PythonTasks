import math
from typing import List
from operator import attrgetter
import cv2 as cv

# чтобы отсеять маленькие объекты типа мелких теней будем учитывать только движущиеся объекты с такой минимальной площадью
MIN_DETECTED_AREA = 8000
ZERO_VELOCITY_THRESHOLD = 0
WHITE = (255, 255, 255)  # белый цвет по умолчанию для рамок движущихся объектов
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WAIT_TIME = 750
HISTORY = 13

# создаем background subtractor для генерирования foreground mask (черно белой маски движущихся объектов)
bg_subtractor = cv.createBackgroundSubtractorMOG2(HISTORY, 128, True)
#bg_subtractor = cv.createBackgroundSubtractorKNN(10,700.0,True)


class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.center = (x + w // 2, y + h // 2)

    def tuple(self):
        return self.x, self.y, self.w, self.h

    def diagonal(self):
        return math.sqrt(self.w**2 + self.h**2)

    def min_side(self):
        return min(self.w, self.h)

class DetectedObjects:
    def __init__(self):
        self.objects: List[DetectedObject] = []

    def find_closest_rect(self, o, rects):
        if not self.objects:
            return None
        rect = min(rects, key = lambda r: math.dist(o.center, r.center))
        #min_dist = math.dist(o.center, rect.center)
        return rect #if min_dist < o.rect.diagonal()/2 + rect.diagonal()/2 else None

    def parse(self, rects):
        def index_to_color(i, o):
            if o.velocity < 0:
                return WHITE
            elif o.velocity <= ZERO_VELOCITY_THRESHOLD:
                return YELLOW
            elif i == 0:
                return RED
            elif i == 1:
                return GREEN
            else:
                return BLUE

        if not self.objects:
            for rect in rects:
                self.objects.append(DetectedObject(rect))
        else:
            for o in self.objects:
                if rects:
                    rect = self.find_closest_rect(o, rects)
                #проблема - а что если прямоугольник принадлжежит новому обьекту
                #есть старые прямоугольники и новые - надо определить какие прямоуголники принадлежат одним и тем же обьектам
                #это можно сделать только путем пересечения прямоуголников - это те прямоугольники которые имеют большую общую площадь
                    o.rect = rect
                    rects.remove(rect)
                else:
                    o.velocity = 0
            for rect in rects:
                self.objects.append(DetectedObject(rect))

        self.objects = sorted(self.objects, key = attrgetter('velocity'), reverse = True)

        for i, o in enumerate(self.objects):
            color = index_to_color(i, o)
            cv.rectangle(frame, o.rect.tuple(), color, 2)

class DetectedObject:
    last_id = 0

    def __init__(self, rect):
        self.id = DetectedObject.last_id
        DetectedObject.last_id += 1
        self.prev_rect = None
        self._rect = rect
        self.velocity = -1
        self.prev_center = None
        self.center = rect.center
        self.color = WHITE

    @property
    def rect(self):
        return self._rect

    @rect.setter
    def rect(self, rect):
        self.prev_rect = self._rect
        self.prev_center = self.center
        self._rect = rect
        self.center = rect.center
        self.velocity = math.dist(self.prev_center, self.center)

def detect_significant_contours_of_motion(mask, threshold_area=400):
    omask = cv.resize(mask, None, fx=0.4, fy=0.4)
    cv.imshow("Foreground oMask", omask)

    _, mask = cv.threshold(mask, 50, 255, cv.THRESH_BINARY) #удаляем шум и тени

    # создаем пиксельное ядро 7х7
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    # расширяем белые пиксели на черно белом кадре для соединения разрозненных элементов движущихся объектов
    mask = cv.dilate(mask, kernel, iterations=6)
    #mask = cv.erode(mask, kernel, iterations=4)
    #mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=4)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #ищем контуры движущихся объектов таких что их площадь больше чем threshold_area (по умолчанию 400 пикселей)
    for contour in contours:
        if cv.contourArea(contour) > threshold_area:
            return True, mask, contours

    return False, mask, contours

#проверяем запускается ли код скрипта напрямую  а не через другой скрипт
if __name__ == '__main__':
    cap = cv.VideoCapture('pool.mp4')

    detected_objects = DetectedObjects()

    frame_counter = 0
    while True:
        #покадрово считываем видео
        ret, frame = cap.read()
        if not ret:
            break

        mask = bg_subtractor.apply(frame)

        frame_counter += 1
        if frame_counter <= HISTORY:
            continue

        #получаем черно белую маску кадра и контуры движущихся объектов
        motion_detected, mask, contours = detect_significant_contours_of_motion(mask, MIN_DETECTED_AREA)

        rects = []

        if motion_detected:
            #  если было движение то проходимся по всем контурам с заданным выше минимальным размером и добавляем
            #  вокруг контуров прямоугольники соответствующего цвета толщины 2
            for contour in contours:
                contour_area = cv.contourArea(contour)
                if contour_area >= MIN_DETECTED_AREA:
                    rects.append(Rectangle(*cv.boundingRect(contour)))
            detected_objects.parse(rects)

        cv.putText(frame, str(int(cap.get(cv.CAP_PROP_POS_FRAMES))), (70, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        # уменьшаем разиер кадров для удобства и выводим их на экран в отдельном окне
        frame = cv.resize(frame, None, fx=0.4, fy=0.4)
        cv.imshow("Original Frame", frame)
        # для отладки можно вывести также в отдельном окне кадры черно белого изображения с маской движущихся объектов
        mask = cv.resize(mask, None, fx=0.4, fy=0.4)
        cv.imshow("Foreground Mask", mask)

        # ждем 30 мс нажатия кнопки q, иначе продолжаем цикл
        if cv.waitKey(WAIT_TIME) & 0xFF == ord('q'):
            break

    # освобождаем все ресурсы выделенные под кадры и окна
    cap.release()
    cv.destroyAllWindows()

