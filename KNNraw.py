import cv2

# создаем background subractor для генерирования foreground mask (черно белой маски движущихся объектов)
# на основе k-Nearest Neighbor (kNN) алгоритма
bg_subtractor = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=700.0, detectShadows=False)

def detect_significant_contours_of_motion(frame, threshold_area=400):
    mask = bg_subtractor.apply(frame)

    # создаем пиксельное ядро 7х7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # расширяем белые пиксели на черно белом кадре для соединения разрозненных элементов движущихся объектов
    mask = cv2.dilate(mask, kernel, iterations=4)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #ищем контуры движущихся объектов таких что их площадь больше чем threshold_area (по умолчанию 400 пикселей)
    for contour in contours:
        if cv2.contourArea(contour) > threshold_area:
            return True, mask, contours

    return False, mask, contours

# чтобы отсеять маленькие объекты типа мелких теней будем учитывать только движущиеся объекты с такой минимальной площадью
min_detected_area = 10000

#проверяем запускается ли код скрипта напрямую  а не через другой скрипт
if __name__ == '__main__':
    cap = cv2.VideoCapture('pool.mp4')

    default_color = (255, 255, 255) #белый цвет по умолчанию для рамок движущихся объектов

    while True:
        #покадрово считываем видео
        ret, frame = cap.read()
        if not ret:
            break

        #получаем черно белую маску кадра и контуры движущихся объектов
        motion_detected, mask, contours = detect_significant_contours_of_motion(frame, min_detected_area)

        if motion_detected:
            #  если было движение то проходимся по всем контурам с заданным выше минимальным размером и добавляем
            #  вокруг контуров прямоугольники белого цвета толщины 2
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area >= min_detected_area:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), default_color, 2)

        # уменьшаем разиер кадров для удобства и выводим их на экран в отдельном окне
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
        cv2.imshow("Original Frame", frame)
        # для отладки можно вывести также в отдельном окне кадры черно белого изображения с маской движущихся объектов
        #mask = cv2.resize(mask, None, fx=0.4, fy=0.4)
        #cv2.imshow("Foreground Mask", mask)

        # ждем 30 мс нажатия кнопки q, иначе продолжаем цикл
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # освобождаем все ресурсы выделенные под кадры и окна
    cap.release()
    cv2.destroyAllWindows()

